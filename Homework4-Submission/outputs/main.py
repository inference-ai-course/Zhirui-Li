"""FastAPI service ¡ª Week 4 RAG agent.

Run with:  uvicorn main:app --reload
Endpoint:  GET /search?q=YOUR+QUESTION
"""
import os, sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent  # outputs/.. = repo root
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from src.llm_client import LLMClient
from src.embeddings import EmbeddingModel
from src.vector_store import FAISSStore
from src.rag_pipeline import RAGPipeline, format_context
from src.rag_evaluation import faithfulness
import src.config as config

INDEX_DIR = ROOT / "outputs" / "project_index"
if not INDEX_DIR.exists():
    raise RuntimeError(f"Index not found at {INDEX_DIR}. Run nb08 first.")

client = LLMClient(path=config.PATH)
em     = EmbeddingModel("all-MiniLM-L6-v2")
store  = FAISSStore.load(str(INDEX_DIR))
rag    = RAGPipeline(em, store, client, retrieve_k=4, rerank_k=4)

app = FastAPI(title="Resume RAG Agent", version="4.0.0")

class Answer(BaseModel):
    question: str
    answer:   str
    sources:  list
    faithfulness: float

@app.get("/search", response_model=Answer)
def search(q: str):
    if not q.strip():
        raise HTTPException(400, "q is required")
    res = rag.answer(q, max_tokens=400)
    f = faithfulness(client, q, res["answer"], res["contexts"])["score"]
    return Answer(question=q, answer=res["answer"], sources=res["sources"], faithfulness=f)

@app.get("/")
def root():
    return {"ok": True, "endpoint": "/search?q=..."}
