# embeddings/embed_upsert_pinecone.py
import os
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configurations
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "finance-faq")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DIM = int(os.getenv("PINECONE_DIM", "384"))

def init_pinecone():
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Using existing Pinecone index: {INDEX_NAME}")

    # Connect to index
    return pc.Index(INDEX_NAME)

def load_chunks(jsonl_path="data/chunks.jsonl"):
    """Load text chunks from a JSONL file."""
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def embed_and_upsert():
    """Generate embeddings and upsert them into Pinecone."""
    print("Loading embedding model:", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    index = init_pinecone()
    chunks = load_chunks()

    BATCH = 32
    print(f"Total chunks to process: {len(chunks)}")

    for i in tqdm(range(0, len(chunks), BATCH)):
        batch = chunks[i:i + BATCH]
        texts = [b["text"] for b in batch]

        # Compute embeddings
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Prepare Pinecone vectors
        to_upsert = []
        for rec, emb in zip(batch, embs):
            meta = {
                "orig_id": rec.get("orig_id"),
                "category": rec.get("category"),
                "sub_category": rec.get("sub_category"),
                "question": rec.get("question"),
                "text": rec.get("text"),
            }
            to_upsert.append((rec["id"], emb.tolist(), meta))

        # Upsert to Pinecone
        index.upsert(vectors=to_upsert)

    print("✅ Upsert complete. Total vectors:", len(chunks))

if __name__ == "__main__":
    embed_and_upsert()
