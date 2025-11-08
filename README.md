# Finance Contextual RAG — Groq + Pinecone + Streamlit

## 1. Overview

Finance Contextual RAG is a complete Retrieval-Augmented Generation (RAG) system designed to answer banking and personal finance questions with reliable, source-grounded responses.  

It combines:
- **Groq LLM** for natural language understanding and generation,
- **Pinecone** for semantic vector storage and retrieval,
- **TF-IDF** for lexical retrieval, and
- **Streamlit** for an interactive chat frontend.

The system is built to be **contextually aware**: each document chunk is first summarized and contextualized by the LLM before being embedded and indexed. This ensures more precise, relevant, and well-grounded retrieval results during question answering.

---

## 2. Contextually-Aware RAG Concept

Traditional RAG pipelines retrieve chunks of text directly from a document store. However, chunks often lose their meaning when taken out of context.

The **contextually-aware RAG** solves this by asking an LLM (Groq) to:
1. Read each chunk of text.
2. Generate a short summary describing what that chunk represents (topic, entities, and purpose).
3. Concatenate that summary with the chunk text before embedding.

This produces more semantically informative vectors, leading to higher-quality retrieval and more accurate answers during inference.

---

## 3. Project Workflow

### A. Indexing Phase (One-Time Setup)

1. Load data files (`.txt`, `.md`, `.json`) from the `data/` folder.
2. Split each document into overlapping chunks (default: 200 words per chunk, 50-word overlap).
3. For each chunk:
   - Generate a **context summary** using the Groq model.
   - Combine the context summary and the original chunk text.
4. Create dense embeddings using the Sentence-Transformers model (`all-MiniLM-L6-v2`).
5. Build a local TF-IDF matrix for word-based (lexical) search.
6. Upload the dense embeddings and metadata to Pinecone.
7. Save the TF-IDF vectorizer and matrix locally in the `embeddings/` folder.

### B. Query Phase (At Runtime)

1. A user asks a question via FastAPI or the Streamlit frontend.
2. The system retrieves:
   - **Dense matches** from Pinecone (semantic similarity).
   - **Sparse matches** from TF-IDF (word-level similarity).
3. Merge both result sets into a **hybrid ranking**.
4. Compose a prompt for Groq:
   - Include the user’s question.
   - Include retrieved text chunks and sources.
   - Optionally include recent chat history.
5. Groq generates a grounded, citation-based answer.

---

---


## 4. Chunking Parameters Explained

| Parameter | Default | Description |
|------------|----------|-------------|
| **Chunk Size** | 200 words | Controls how much text goes into one embedding unit. Large enough to capture context, small enough to fit within model limits. |
| **Overlap** | 50 words | Keeps continuity between chunks, ensuring no important sentence or meaning is cut off. |

This combination (200 words + 50 overlap) balances **semantic clarity** with **retrieval precision**.

---

## 5. Sparse Matrix and TF-IDF (Simplified Explanation)

The TF-IDF index represents all chunks as a large table of word importance:

- **TF (Term Frequency):** How often a word appears in a chunk.  
- **IDF (Inverse Document Frequency):** How unique that word is across all chunks.  

The TF-IDF values are stored in a **sparse matrix** (mostly zeros), where:
- Rows = chunks
- Columns = words
- Cell = importance score

This matrix is saved locally for fast word-based retrieval. When a user asks a question, the system converts that question into the same TF-IDF form and quickly finds which chunks share similar words.

---

## 6. API Design

### Endpoints

| Method | Route | Description |
|--------|--------|-------------|
| `GET` | `/health` | Health check endpoint |
| `POST` | `/query` | Standard retrieval + answer generation |
| `POST` | `/chat` | Conversational endpoint with chat memory |

### Example Request (to `/query`)

```json
{
  "query": "Explain the difference between credit score and CIBIL score",
  "k": 5
}

{
  "query": "Explain the difference between credit score and CIBIL score",
  "results": [...],
  "answer": "A CIBIL score is one type of credit score specific to India, while credit score is a general term used globally..."
}

streamlit run ui/streamlit_chat.py

.env file in the project root:(Excluded due to git policy)
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=finance-faq
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=200
CHUNK_OVERLAP=50
NUM_RESULTS=5

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000

Running Steps

cd D:\Inheritance\finance_contextual_rag_groq_pinecone
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Step 2 — Add Data

Place your finance-related text or JSON files inside:

data/


Example:
data/Finance_FAQ_Extended.json or data/finance_faq.txt

Step 3 — Build the Index
python -m indexer.contextual_indexer_faq

Step 4 — Start the Backend
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload


Open http://localhost:8000/docs
 to view the API.

Step 5 — Start the Frontend
streamlit run ui/streamlit_chat.py


Access the UI at http://localhost:8501
.


##  SYSTEM DESIGN
+------------------+      +--------------------+      +----------------+
|   Text Data      | ---> | Chunk & Contextual | ---> | Embedding Model|
| (FAQ Documents)  |      | Summary (Groq)     |      | (MiniLM)       |
+------------------+      +--------------------+      +----------------+
                                                           |
                                                           v
                                      +-----------------------------+
                                      |   Pinecone Vector Database   |
                                      | (Dense Semantic Retrieval)   |
                                      +-----------------------------+
                                               |
                                               v
                                      +-----------------------------+
                                      |    TF-IDF Sparse Index      |
                                      | (Lexical Retrieval)         |
                                      +-----------------------------+
                                               |
                                               v
                                      +-----------------------------+
                                      |  Hybrid Search (Dense+Sparse)|
                                      +-----------------------------+
                                               |
                                               v
                                      +-----------------------------+
                                      |  Groq LLM Answer Generation  |
                                      |  (Grounded in Sources)       |
                                      +-----------------------------+
                                               |
                                               v
                                      +-----------------------------+
                                      |  Streamlit Chat Frontend     |
                                      +-----------------------------+


----
