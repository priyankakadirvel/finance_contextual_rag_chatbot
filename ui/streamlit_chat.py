import streamlit as st
import os
from retriever.contextual_retriever import hybrid_search
from groq import Groq

# Load secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")

# Groq Client
client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query, passages, history=[]):
    # Prepare context
    context_blob = "\n\n".join(
        [f"[{i+1}] {p['meta'].get('source','')} :: {p['chunk']}" for i,p in enumerate(passages)]
    )

    # History formatting
    history_text = ""
    if history:
        history_text = "\n".join([f"{h['role'].capitalize()}: {h['text']}" for h in history[-3:]])

    system_prompt = "You are a finance RAG assistant. Use ONLY the provided sources. Cite as [1],[2], etc."

    user_prompt = f"""
Chat History:
{history_text}

User Query: {query}

Relevant Sources:
{context_blob}

Answer clearly with citations.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    return response.choices[0].message.content

# Streamlit UI
st.title("Finance RAG Chatbot (Groq + Pinecone)")

if "history" in st.session_state:
    history = st.session_state["history"]
else:
    st.session_state["history"] = []
    history = []

user_query = st.text_input("Ask a question:")

if st.button("Send"):
    if user_query.strip():
        passages = hybrid_search(user_query, k=5)
        answer = generate_answer(user_query, passages, st.session_state["history"])

        st.session_state["history"].append({"role": "user", "text": user_query})
        st.session_state["history"].append({"role": "assistant", "text": answer})

# Show chat
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")
