"""
streamlit_app.py - Streamlit chat UI for the RAG Document Q&A system.

Features
--------
- Sidebar: PDF file uploader (calls /upload) and "View Logs" expander.
- Main area: chat-style interface with conversation history.
- Source citations displayed in a collapsed expander below each answer.
"""

import os

import requests
import streamlit as st
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to add it to the knowledge base.",
    )

    if uploaded_file is not None:
        if st.button("📤 Ingest PDF", use_container_width=True):
            with st.spinner("Uploading and indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
                    if resp.status_code == 201:
                        data = resp.json()
                        st.success(f"✅ {data['message']}")
                    else:
                        st.error(f"❌ {resp.json().get('detail', 'Upload failed')}")
                except requests.ConnectionError:
                    st.error("⚠️ Cannot reach the API. Is the FastAPI server running?")

    st.divider()

    with st.expander("📊 View Query Logs"):
        if st.button("Refresh Logs", use_container_width=True):
            try:
                resp = requests.get(f"{API_BASE}/logs", timeout=10)
                if resp.status_code == 200:
                    logs = resp.json()
                    if logs:
                        df = pd.DataFrame(logs)
                        df = df[["id", "timestamp", "query", "answer", "latency_ms"]]
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No logs yet.")
                else:
                    st.error("Failed to fetch logs.")
            except requests.ConnectionError:
                st.error("⚠️ Cannot reach the API.")

# ── Main area ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    "<h1>📄 RAG Document Q&A</h1>"
    "<p>Upload PDFs and ask questions — answers are grounded with source citations.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Source Citations"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Chunk {i}** — *{src['source']}*, Page {src['page']} "
                        f"(score: {src['score']})"
                    )
                    st.code(src["content"][:300], language=None)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ask",
                    json={"query": prompt},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Source Citations"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(
                                    f"**Chunk {i}** — *{src['source']}*, Page {src['page']} "
                                    f"(score: {src['score']})"
                                )
                                st.code(src["content"][:300], language=None)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    err = resp.json().get("detail", "Something went wrong.")
                    st.error(f"❌ {err}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {err}"}
                    )
            except requests.ConnectionError:
                st.error("⚠️ Cannot reach the API. Is the FastAPI server running on port 8000?")
