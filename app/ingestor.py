"""
ingestor.py — PDF loading, chunking, embedding, and FAISS indexing.

Responsibilities
----------------
1. Extract text from uploaded PDFs page-by-page using PyMuPDF.
2. Split text into overlapping chunks with LangChain's
   RecursiveCharacterTextSplitter.
3. Embed chunks using the all-MiniLM-L6-v2 sentence-transformer model.
4. Persist a FAISS index to disk; if one already exists, merge new
   documents into it rather than overwriting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ── Defaults ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vector_store",
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ── PDF extraction ──────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """
    Open a PDF with PyMuPDF and extract text page-by-page.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    list[Document]
        One LangChain Document per page, with ``source`` and ``page``
        metadata.
    """
    documents: List[Document] = []
    filename = Path(pdf_path).name

    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": filename, "page": page_num},
                        )
                    )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read PDF '{pdf_path}': {exc}"
        ) from exc

    return documents


# ── Chunking ────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split page-level documents into smaller overlapping chunks.

    Parameters
    ----------
    documents : list[Document]
        Page-level documents from ``extract_text_from_pdf``.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    list[Document]
        Chunked documents with inherited metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ── FAISS indexing ──────────────────────────────────────────────────────

def build_or_update_index(
    chunks: List[Document],
    persist_dir: str = VECTOR_STORE_DIR,
) -> int:
    """
    Embed *chunks* and either create a new FAISS index or merge into
    an existing one.

    Parameters
    ----------
    chunks : list[Document]
        Chunked documents to index.
    persist_dir : str
        Directory where the FAISS index is persisted.

    Returns
    -------
    int
        Total number of chunks now in the index.
    """
    embeddings = _get_embeddings()
    index_path = os.path.join(persist_dir, "index.faiss")

    if os.path.exists(index_path):
        # Load existing index and merge new chunks
        existing_store = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        new_store = FAISS.from_documents(chunks, embeddings)
        existing_store.merge_from(new_store)
        existing_store.save_local(persist_dir)
        total = len(existing_store.docstore._dict)
    else:
        os.makedirs(persist_dir, exist_ok=True)
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(persist_dir)
        total = len(store.docstore._dict)

    return total


# ── Public convenience function ─────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> int:
    """
    End-to-end pipeline: extract → chunk → embed → index.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF to ingest.

    Returns
    -------
    int
        Number of chunks produced from the uploaded PDF.
    """
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_documents(pages)
    build_or_update_index(chunks)
    return len(chunks)
