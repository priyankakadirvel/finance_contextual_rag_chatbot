import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Hybrid search using ONLY Pinecone (no embedding model)
def hybrid_search(query, k=5):
    response = index.query(
        query=query,
        top_k=k,
        include_metadata=True
    )

    results = []
    for match in response["matches"]:
        results.append({
            "chunk": match["metadata"].get("chunk", ""),
            "score": match["score"],
            "meta": match["metadata"]
        })
    return results
