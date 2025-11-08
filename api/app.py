# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from retriever.contextual_retriever import hybrid_search
from groq import Groq
from typing import List, Dict, Any

load_dotenv()

app = FastAPI(title="Contextual RAG (Groq + Pinecone)")

# ===== Models =====
class QueryRequest(BaseModel):
    query: str
    k: int = int(os.getenv("NUM_RESULTS", "5"))

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []
    session_meta: Dict[str, Any] = {}
    top_k: int = int(os.getenv("NUM_RESULTS", "5"))

# ===== Helper: Groq answer =====
def groq_answer(query: str, passages: list, history: List[Dict[str, str]] = None) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Include chat history as context (last 3 exchanges)
    history_context = ""
    if history:
        history_context = "\n".join(
            [f"{t['role'].capitalize()}: {t['text']}" for t in history[-3:]]
        )

    # Prepare retrieved context
    context_blob = "\n\n".join(
        [f"[{i+1}] {p['meta'].get('source','')} :: {p['chunk']}" for i, p in enumerate(passages)]
    )

    system = "You are a helpful finance RAG assistant. Use only the provided sources. Cite sources as [1], [2], etc."
    user = f"Chat History:\n{history_context}\n\nUser Query: {query}\n\nRelevant Sources:\n{context_blob}\n\nAnswer with citations and clarity:"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()

# ===== Routes =====
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query")
async def query(q: QueryRequest):
    if not q.query or not q.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    passages = hybrid_search(q.query, k=q.k)
    answer = groq_answer(q.query, passages)
    return {"query": q.query, "results": passages, "answer": answer}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    passages = hybrid_search(req.query, k=req.top_k)
    answer = groq_answer(req.query, passages, history=req.history)

    return {
        "query": req.query,
        "answer": answer,
        "results": passages,
        "history": req.history + [{"role": "user", "text": req.query}, {"role": "assistant", "text": answer}],
    }
