"""
api.py - FastAPI application with /upload, /ask, and /logs endpoints.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.generator import ask as generate_answer
from app.ingestor import ingest_pdf
from app.monitor import get_logs, log_query

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload PDFs and ask questions. Answers are grounded with citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(DATA_DIR, exist_ok=True)


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    query: str


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""
    answer: str
    sources: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    """Response body for the /upload endpoint."""
    filename: str
    chunk_count: int
    message: str


@app.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a PDF upload, persist it, and run the ingestion pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    save_path = os.path.join(DATA_DIR, file.filename)
    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    try:
        chunk_count = ingest_pdf(save_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return UploadResponse(
        filename=file.filename,
        chunk_count=chunk_count,
        message=f"Ingested '{file.filename}' into {chunk_count} chunks.",
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest) -> AskResponse:
    """Run retriever + generator, log the interaction, return answer."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    start = time.perf_counter()
    try:
        result = generate_answer(body.query)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    latency_ms = (time.perf_counter() - start) * 1000
    try:
        log_query(body.query, result["sources"], result["answer"], latency_ms)
    except Exception:
        pass  # Logging failure should not break the response

    return AskResponse(answer=result["answer"], sources=result["sources"])


@app.get("/logs")
async def fetch_logs() -> List[Dict[str, Any]]:
    """Return the 20 most recent query log entries."""
    try:
        return get_logs(limit=20)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {exc}")
