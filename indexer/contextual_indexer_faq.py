# indexer/contextual_indexer_faq.py
import os, json, time, hashlib
from pathlib import Path
from dotenv import load_dotenv
from preprocessing.chunker import simple_split_text
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
from tqdm import tqdm

# Groq
from groq import Groq

# Pinecone v3
from pinecone import Pinecone, ServerlessSpec

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

DOCS_DIR = "./data"
OUT_DIR = "./embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

# Config
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "finance-contextual-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

def load_text_files(directory: str):
    files = []
    for p in Path(directory).glob("**/*"):
        if p.is_file() and p.suffix.lower() in [".txt",".md",".text"]:
            files.append(p)
    return files

def groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)

def situate_chunk_groq(client: Groq, chunk: str, filename: str) -> str:
    prompt = f"""
You will be given a chunk from a document (filename: {filename}).
Produce a short contextual description (around 80–150 tokens) describing where this chunk sits in the document,
its purpose, and any helpful metadata (topic, entities). Be concise and factual.

Chunk:
\"\"\"
{chunk}
\"\"\"
Context:
"""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=180,
    )
    return resp.choices[0].message.content.strip()

def ensure_pinecone_index(pc: Pinecone, index_name: str, dim: int):
    names = [i.name for i in pc.list_indexes()]
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

def build_and_upload():
    # 1) Chunk + Context (Groq)
    files = load_text_files(DOCS_DIR)
    if not files:
        print("No files found in ./data. Add .txt/.md files and rerun.")
        return

    gclient = groq_client()
    embedder = SentenceTransformer(EMBED_MODEL)

    contextualized = []
    metadata = []

    print("Building contextual chunks via Groq...")
    for f in tqdm(files):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        chunks = simple_split_text(raw, max_words=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for idx, ch in enumerate(chunks):
            ctx = situate_chunk_groq(gclient, ch, f.name)
            text = (ctx + "\n\n" + ch).strip()
            contextualized.append(text)
            metadata.append({"source": str(f), "chunk_idx": idx, "context": ctx})

    # Persist chunks & metadata locally (for TF-IDF / audit)
    with open(os.path.join(OUT_DIR, "chunks.json"), "w", encoding="utf-8") as fh:
        json.dump(contextualized, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    # 2) TF-IDF (local for hybrid)
    print("Building TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(contextualized)
    joblib.dump(vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(OUT_DIR, "tfidf_matrix.joblib"))
    print("TF-IDF saved.")

    # 3) Pinecone upload
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    print("Encoding embeddings...")
    vecs = embedder.encode(contextualized, show_progress_bar=True, convert_to_numpy=True)
    dim = vecs.shape[1]

    ensure_pinecone_index(pc, PINECONE_INDEX_NAME, dim)
    index = pc.Index(PINECONE_INDEX_NAME)

    # upsert in batches
    print("Upserting to Pinecone...")
    batch = 100
    for i in tqdm(range(0, len(contextualized), batch)):
        ids = []
        up = []
        for j, text in enumerate(contextualized[i:i+batch], start=i):
            _id = hashlib.md5(f"{metadata[j]['source']}::{metadata[j]['chunk_idx']}".encode()).hexdigest()
            ids.append(_id)
            up.append({
                "id": _id,
                "values": vecs[j].tolist(),
                "metadata": {
                    "text": text,
                    "source": metadata[j]["source"],
                    "chunk_idx": metadata[j]["chunk_idx"],
                }
            })
        index.upsert(vectors=up)
    print("Pinecone upsert complete.")

def main():
    build_and_upload()
    print("Indexing complete.")

if __name__ == "__main__":
    main()
