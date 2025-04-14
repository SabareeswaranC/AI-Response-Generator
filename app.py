import os
import re
import unicodedata
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from flask_cors import CORS

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ---------------------- Flask app and model setup ----------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_pipeline = pipeline("sentiment-analysis")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API Key. Please provide OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o")

df = pd.read_csv("chatbot-faqs.csv")

# ---------------------- Preprocessing Function ----------------------

def clean_text(text):
    text = unicodedata.normalize("NFKD", text) 
    text = re.sub(r'[^\x20-\x7E\u0080-\uFFFF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["Cleaned Query"] = df["User Query"].astype(str).apply(clean_text)

# ---------------------- FAISS Index Handling ----------------------

try:
    with open("faq_index.pkl", "rb") as f:
        faiss_index, df = pickle.load(f)
except FileNotFoundError:
    embeddings = model.encode(df["Cleaned Query"].tolist())
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    with open("faq_index.pkl", "wb") as f:
        pickle.dump((faiss_index, df), f)

# ---------------------- Utility Functions ----------------------

def get_tone_instruction(sentiment, rating):
    if (rating >= 4 and sentiment == "negative") or (rating <= 3 and sentiment == "positive"):
        return "Generate a response that balances appreciation and constructive acknowledgment in two lines."
    elif rating >= 4 and sentiment == "positive":
        return "Generate a helpful, friendly response thanking the user and showing appreciation in two lines."
    else:
        return "Generate a helpful, friendly response acknowledging the issue and showing empathy in two lines."

def get_prompt_template(rating: int, sentiment: str, matched_q: str = "", matched_a: str = ""):
    has_faq = pd.notna(matched_q) and pd.notna(matched_a) and matched_q.strip() and matched_a.strip()

    if has_faq:
        template = """
A user left a {sentiment} review: "{user_input}" with a rating of {rating} stars.

Please rephrase the below response: 
{matched_answer}

And should not ask them to give good review or rating.

Generate a helpful and accurate response. 
- Be empathic and conversational.
- Clearly explain exactly.
- Only suggest contacting support if the response don’t fully resolve the issue.
- if needed only provide Contact details: care@zaggle.in

"""
    else:
        template = """
A user left a {sentiment} review: "{user_input}" with a rating of {rating} stars.

{tone_instruction}

And should not ask them to give good review or rating.

- Use the user's input to empathize response.
- Only suggest contacting support if the issue cannot be resolved in the reply.
- If needed, include: Contact - care@zaggle.in

"""

    return PromptTemplate.from_template(template)

def get_sentiment_label(text):
    cleaned_text = clean_text(text)
    result = sentiment_pipeline(cleaned_text)[0]
    return result["label"].lower()

def find_similar_question(user_question):
    cleaned_question = clean_text(user_question)
    user_embedding = model.encode([cleaned_question])
    D, I = faiss_index.search(np.array(user_embedding), k=1)
    similarity_score = 1 - D[0][0]

    if similarity_score < 0.65:
        return "", ""
    
    matched_idx = I[0][0]
    return (
        df.iloc[matched_idx]["User Query"],
        df.iloc[matched_idx]["Product Responses"]
    )

# ---------------------- Flask Route ----------------------

user_reviews = []

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    sentiment = ""
    rating = 0
    tone_instruction = ""
    if request.method == "POST":
        user_input = request.form.get("review", "").strip()
        rating = int(request.form.get("rating", 0))
        if user_input or rating:
            try:
                sentiment = get_sentiment_label(user_input)
                matched_q, matched_a = find_similar_question(user_input)
                prompt = get_prompt_template(
                    rating,
                    sentiment,
                    matched_q,
                    matched_a
                )
                chain = prompt | llm
                tone_instruction = get_tone_instruction(sentiment, rating)

                output = chain.invoke({
                    "user_input": user_input,
                    "matched_question": matched_q,
                    "matched_answer": matched_a,
                    "sentiment": sentiment,
                    "rating": rating,
                    "tone_instruction": tone_instruction
                })

                response = output.content.strip()

                user_reviews.append({
                    'input': user_input,
                    'rating': rating,
                    'response': response
                })

                user_reviews.reverse()
            except Exception as e:
                response = f"An error occurred: {str(e)}"
    return render_template("index.html", response=response, sentiment=sentiment, rating=rating, user_reviews=user_reviews)

# ---------------------- App Entry ----------------------

if __name__ == "__main__":
    app.run(debug=True)
