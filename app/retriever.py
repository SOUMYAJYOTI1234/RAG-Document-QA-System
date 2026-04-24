"""
retriever.py — FAISS similarity search and top-k chunk retrieval.

Responsibilities
----------------
1. Load the persisted FAISS index from disk.
2. Accept a natural-language query, embed it, and return the top-k most
   similar document chunks.
3. Return chunks together with their source document name and page number.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.ingestor import EMBEDDING_MODEL, VECTOR_STORE_DIR

# ── Defaults ────────────────────────────────────────────────────────────
TOP_K: int = 5


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_index(persist_dir: str = VECTOR_STORE_DIR) -> FAISS:
    """
    Load the persisted FAISS index.

    Parameters
    ----------
    persist_dir : str
        Directory that contains ``index.faiss`` and ``index.pkl``.

    Returns
    -------
    FAISS
        A LangChain FAISS vector store ready for querying.

    Raises
    ------
    FileNotFoundError
        If the index files do not exist on disk.
    """
    index_path = os.path.join(persist_dir, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No FAISS index found at '{persist_dir}'. "
            "Please upload and ingest at least one PDF first."
        )

    return FAISS.load_local(
        persist_dir,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
    persist_dir: str = VECTOR_STORE_DIR,
) -> List[Dict[str, Any]]:
    """
    Embed *query* and return the top-k most similar chunks.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    top_k : int
        Number of chunks to retrieve.
    persist_dir : str
        Directory of the persisted FAISS index.

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``content``: the chunk text
        - ``source``: originating PDF filename
        - ``page``: page number in the PDF
        - ``score``: similarity score (lower = more similar for L2)
    """
    store = load_index(persist_dir)
    results = store.similarity_search_with_score(query, k=top_k)

    chunks: List[Dict[str, Any]] = []
    for doc, score in results:
        chunks.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "score": round(float(score), 4),
            }
        )

    return chunks


def get_retriever(
    top_k: int = TOP_K,
    persist_dir: str = VECTOR_STORE_DIR,
):
    """
    Return a LangChain-compatible retriever backed by the FAISS index.

    Parameters
    ----------
    top_k : int
        Number of results to return per query.
    persist_dir : str
        Directory of the persisted FAISS index.

    Returns
    -------
    langchain.vectorstores.base.VectorStoreRetriever
        A retriever that can be plugged into a LangChain chain.
    """
    store = load_index(persist_dir)
    return store.as_retriever(search_kwargs={"k": top_k})
