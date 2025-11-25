# ui/streamlit_chat.py
import streamlit as st
import requests
from dotenv import load_dotenv
import os, json

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Finance Chatbot", page_icon="💬")
st.title("Finance Chatbot — Banking & Personal Finance")

if "history" not in st.session_state:
    st.session_state.history = []

def send_query(query):
    payload = {"query": query, "history": st.session_state.history, "session_meta": {}, "top_k": 5}
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        return data
    else:
        st.error(f"API error: {resp.status_code} {resp.text}")
        return None

with st.form("user_form", clear_on_submit=True):
    user_input = st.text_input("Ask your banking / personal finance question")
    submitted = st.form_submit_button("Send")
    if submitted and user_input.strip():
        # append user turn
        st.session_state.history.append({"role":"user", "text": user_input})
        result = send_query(user_input)
        if result:
            # append assistant turn
            st.session_state.history.append({"role":"assistant", "text": result["answer"]})
            st.rerun()

st.write("---")
for turn in st.session_state.history[::-1]:
    if turn["role"] == "user":
        st.markdown(f"**You:** {turn['text']}")
    else:
        st.markdown(f"**Assistant:** {turn['text']}")
