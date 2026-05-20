from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Optional Gemini integration via Google Generative AI
import google.generativeai as genai

app = FastAPI()

# Allow cross‑origin requests from any origin. In production you may restrict this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model once at startup on CPU
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Define a small corpus of documents as a demonstration. Replace with your data.
documents = [
    "Le RAG combine recherche d'information et generation de texte.",
    "Les embeddings transforment un texte en vecteur numerique.",
    "Le chunking consiste a decouper un document en petits morceaux.",
    "Le retrieval permet de retrouver les passages les plus pertinents.",
    "Un serveur GPU permet d'accelerer certains calculs de machine learning."
]

# Precompute embeddings for all documents to speed up queries
doc_embeddings = embedding_model.encode(documents)

# Data model for incoming question requests
class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "API RAG fonctionne"}

@app.post("/ask")
def ask_question(data: QuestionRequest):
    """
    Receive a question, find the most relevant document using cosine similarity,
    optionally call Gemini to generate an answer, and return the result.
    """
    question = data.question

    # Step 1: Compute embedding for the question
    question_embedding = embedding_model.encode([question])

    # Step 2: Compute cosine similarity scores with all documents
    scores = cosine_similarity(question_embedding, doc_embeddings)[0]

    # Step 3: Select the highest scoring document
    best_index = int(np.argmax(scores))
    best_document = documents[best_index]
    best_score = float(scores[best_index])

    # Fallback answer based on retrieved context only
    fallback_answer = (
        "D'apres le contexte retrouve, " + best_document
    )

    # Check for Gemini API key and attempt generation
    api_key = os.getenv("GEMINI_API_KEY")
    answer = fallback_answer + " // Gemini non configure."

    if api_key:
        try:
            genai.configure(api_key=api_key)
            prompt = f"""
Reponds uniquement avec le contexte fourni.
Si le contexte ne suffit pas, dis que tu ne sais pas.

Question :
{question}

Contexte :
{best_document}
"""
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            # In case of errors with Gemini, fall back to context-based answer
            answer = fallback_answer + f" // Gemini indisponible : {e}"

    return {
        "question": question,
        "answer": answer,
        "context": best_document,
        "score": best_score,
    }