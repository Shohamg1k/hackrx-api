import json
import requests
import fitz  # PyMuPDF
import nltk
import uuid
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

nltk.download("punkt")

# ---- Authentication ----
API_TOKEN = "3c919b8dc957731a1c162c3d36b1fa8c3cf3d40adeb5e10fca649729a3bb4eaf"

# ---- LLM + Embeddings ----
llm = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- FastAPI Setup ----
app = FastAPI()

# ---- Data Model ----
class QARequest(BaseModel):
    documents: str
    questions: list[str]

# ---- Helper: Download PDF ----
def download_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download PDF.")
    return response.content

# ---- Helper: Extract Text ----
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# ---- Helper: Chunking ----
def chunk_text(text, chunk_size=400):
    sentences = nltk.sent_tokenize(text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

# ---- Helper: Clause Matching ----
def find_relevant_chunks(question, chunks):
    question_emb = embedder.encode(question, convert_to_tensor=True)
    chunk_embs = embedder.encode(chunks, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_emb, chunk_embs)[0]
    top_k = scores.topk(2)
    return [chunks[i] for i in top_k.indices]

# ---- Helper: LLM Answering ----
def query_llm(question, context):
    prompt = f"Context:\n{context}\n\nAnswer the question precisely:\n{question}"
    response = llm(prompt, max_new_tokens=150, do_sample=False)
    return response[0]["generated_text"].split("Answer the question precisely:")[-1].strip()

# ---- Main Run Logic ----
def process_questions(pdf_url, questions):
    pdf = download_pdf_from_url(pdf_url)
    text = extract_text_from_pdf(pdf)
    chunks = chunk_text(text)

    answers = []
    for q in questions:
        relevant_chunks = find_relevant_chunks(q, chunks)
        combined_context = " ".join(relevant_chunks)
        answer = query_llm(q, combined_context)
        answers.append(answer)

    return answers

# ---- Endpoint: /hackrx/run ----
@app.post("/api/v1/hackrx/run")
def run_qa(request: QARequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        answers = process_questions(request.documents, request.questions)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
