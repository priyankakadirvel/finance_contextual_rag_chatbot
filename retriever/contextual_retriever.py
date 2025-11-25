# retriever/contextual_retriever.py
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Hybrid search using only Pinecone (no sentence-transformers)
def hybrid_search(query, k=5):
    # query is string → Pinecone will use its server-side sparse/dense hybrid search
    # NOTE: you MUST have stored embeddings already during ingestion
    response = index.query(
        top_k=k,
        include_metadata=True,
        vector=None,        # No sentence-transformer embeddings
        query=query,        # Use Pinecone text search mode
    )

    results = []
    for match in response["matches"]:
        results.append({
            "chunk": match["metadata"].get("chunk", ""),
            "score": match["score"],
            "meta": match["metadata"]
        })
    return results
