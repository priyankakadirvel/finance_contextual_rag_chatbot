# retriever/contextual_retriever.py
import os, json, joblib, numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

OUT_DIR = "./embeddings"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "finance-contextual-rag")

EMBEDDER = SentenceTransformer(EMBED_MODEL)

# local artifacts for TF-IDF hybrid
with open(os.path.join(OUT_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)
with open(os.path.join(OUT_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    METADATA = json.load(f)
TFIDF_VECT = joblib.load(os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
TFIDF_MAT = joblib.load(os.path.join(OUT_DIR, "tfidf_matrix.joblib"))

# Pinecone client
PC = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX = PC.Index(INDEX_NAME)

def pinecone_search(query: str, k: int = 5):
    q = EMBEDDER.encode([query], convert_to_numpy=True)[0]
    res = INDEX.query(vector=q.tolist(), top_k=k, include_metadata=True)
    out = []
    for m in res.matches:
        out.append({
            "chunk": m.metadata.get("text",""),
            "score": float(m.score),
            "meta": {
                "source": m.metadata.get("source",""),
                "chunk_idx": m.metadata.get("chunk_idx",-1)
            }
        })
    return out

def tfidf_search(query: str, k: int = 5):
    q_vec = TFIDF_VECT.transform([query])
    sims = (TFIDF_MAT @ q_vec.T).toarray().squeeze()
    idxs = sims.argsort()[::-1][:k]
    return [{
        "chunk": CHUNKS[i],
        "score": float(sims[i]),
        "meta": METADATA[i]
    } for i in idxs]

def hybrid_search(query: str, k: int = 5):
    a = pinecone_search(query, k=k*2)
    b = tfidf_search(query, k=k*2)
    # combine by chunk text
    scores = {}
    for r in a + b:
        key = r["chunk"]
        scores.setdefault(key, []).append(r["score"])
    merged = [{"chunk": c, "score": sum(v)/len(v)} for c, v in scores.items()]
    merged = sorted(merged, key=lambda x: x["score"], reverse=True)[:k]
    # attach metadata
    results = []
    for m in merged:
        # try local metadata first
        if m["chunk"] in CHUNKS:
            idx = CHUNKS.index(m["chunk"])
            meta = METADATA[idx]
        else:
            meta = {"source":"pinecone","chunk_idx":-1}
        results.append({"chunk": m["chunk"], "score": m["score"], "meta": meta})
    return results
