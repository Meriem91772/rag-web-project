import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

# Autoriser le frontend à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Corpus simple pour ton mini RAG
documents = [
    "Le RAG combine la recherche d'information et la génération de texte.",
    "Les embeddings transforment un texte en vecteur numérique.",
    "Le chunking consiste à découper un document en petits morceaux.",
    "Le retrieval permet de retrouver les passages les plus pertinents.",
    "Un serveur GPU permet d'accélérer certains calculs de machine learning.",
    "Pour réduire les hallucinations dans un système RAG, il faut utiliser un contexte fiable, limiter la réponse aux documents retrouvés et citer les sources."
]

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "message": "API RAG fonctionne sur Render"}


def simple_retrieval(question: str):
    question_words = set(question.lower().split())

    best_doc = documents[0]
    best_score = 0

    for doc in documents:
        doc_words = set(doc.lower().split())
        score = len(question_words.intersection(doc_words))

        if score > best_score:
            best_score = score
            best_doc = doc

    return best_doc, best_score


@app.post("/ask")
def ask_question(data: QuestionRequest):
    question = data.question

    best_document, score = simple_retrieval(question)

    fallback_answer = f"D'après le contexte retrouvé : {best_document}"

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "question": question,
            "answer": fallback_answer + " // Gemini non configuré.",
            "context": best_document,
            "score": score
        }

    try:
        genai.configure(api_key=api_key)

        prompt = f"""
Tu es un assistant RAG.
Réponds uniquement à partir du contexte fourni.
Si le contexte ne suffit pas, dis clairement que tu ne sais pas.

Question :
{question}

Contexte :
{best_document}
"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        return {
            "question": question,
            "answer": response.text,
            "context": best_document,
            "score": score
        }

    except Exception as e:
        return {
            "question": question,
            "answer": fallback_answer + f" // Erreur Gemini : {str(e)}",
            "context": best_document,
            "score": score
        }
