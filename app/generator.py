"""
generator.py — LangChain chain: retrieved context + Groq LLM answer.

Responsibilities
----------------
1. Build a RetrievalQA-style chain using Groq's llama3-8b-8192 model.
2. Use a custom prompt that forces the LLM to:
   a. Answer ONLY from the provided context.
   b. Say "I don't know" when the context lacks the answer.
   c. Always cite source document and page number.
3. Return both the answer and the retrieved source chunks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from app.retriever import get_retriever, retrieve_chunks

load_dotenv()

# ── Prompt ──────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """\
You are a precise, helpful document assistant. Use ONLY the context below
to answer the user's question. Follow these rules strictly:

1. Answer ONLY from the provided context — do NOT use prior knowledge.
2. If the answer is NOT in the context, respond exactly:
   "I don't know based on the provided documents."
3. Always cite the source document name and page number for every claim
   using the format: [Source: <filename>, Page: <page>].
4. Be concise and well-structured.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# ── LLM ─────────────────────────────────────────────────────────────────
MODEL_NAME: str = "llama3-8b-8192"


def _get_llm() -> ChatGroq:
    """
    Instantiate the Groq LLM.

    Returns
    -------
    ChatGroq
        A LangChain-compatible Groq chat model.

    Raises
    ------
    ValueError
        If the ``GROQ_API_KEY`` environment variable is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    return ChatGroq(
        model_name=MODEL_NAME,
        api_key=api_key,
        temperature=0,
        max_tokens=1024,
    )


# ── Chain construction ──────────────────────────────────────────────────

def build_qa_chain() -> RetrievalQA:
    """
    Build and return a LangChain RetrievalQA chain backed by FAISS + Groq.

    Returns
    -------
    RetrievalQA
        Ready-to-invoke chain.
    """
    llm = _get_llm()
    retriever = get_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain


# ── Public entry point ──────────────────────────────────────────────────

def ask(query: str) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for a user query.

    Parameters
    ----------
    query : str
        Natural-language question about the ingested documents.

    Returns
    -------
    dict
        ``answer`` — the LLM-generated response.
        ``sources`` — list of dicts with ``content``, ``source``,
        ``page``, and ``score`` for each retrieved chunk.
    """
    # Retrieve chunks with scores (for logging / API response)
    source_chunks: List[Dict[str, Any]] = retrieve_chunks(query)

    # Run the chain
    chain = build_qa_chain()
    result = chain.invoke({"query": query})

    return {
        "answer": result["result"],
        "sources": source_chunks,
    }
