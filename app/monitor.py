"""
monitor.py — SQLite logging for query tracking and experiment monitoring.

Responsibilities
----------------
1. Create / open a SQLite database at ``logs/query_logs.db``.
2. Maintain a ``query_logs`` table with:
   id, timestamp, query, retrieved_chunks, answer, latency_ms.
3. Expose ``log_query()`` — called after every ``/ask`` request.
4. Expose ``get_logs()`` — returns the most recent log entries.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

# ── Defaults ────────────────────────────────────────────────────────────
LOGS_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
)
DB_PATH: str = os.path.join(LOGS_DIR, "query_logs.db")


def _get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and ensure the table exists.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        An open database connection.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            query         TEXT    NOT NULL,
            retrieved_chunks TEXT NOT NULL,
            answer        TEXT    NOT NULL,
            latency_ms    REAL   NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def log_query(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    answer: str,
    latency_ms: float,
    db_path: str = DB_PATH,
) -> None:
    """
    Insert a single query log entry.

    Parameters
    ----------
    query : str
        The user's question.
    retrieved_chunks : list[dict]
        Chunks returned by the retriever (serialised as JSON).
    answer : str
        The LLM-generated answer.
    latency_ms : float
        End-to-end latency for the ``/ask`` request in milliseconds.
    db_path : str
        Path to the SQLite database file.
    """
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO query_logs
                (timestamp, query, retrieved_chunks, answer, latency_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                query,
                json.dumps(retrieved_chunks, ensure_ascii=False),
                answer,
                round(latency_ms, 2),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_logs(
    limit: int = 20,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent query log entries.

    Parameters
    ----------
    limit : int
        Maximum number of entries to return (newest first).
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    list[dict]
        Each dict mirrors a row in the ``query_logs`` table.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, timestamp, query, retrieved_chunks, answer, latency_ms
            FROM query_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    logs: List[Dict[str, Any]] = []
    for row in rows:
        logs.append(
            {
                "id": row[0],
                "timestamp": row[1],
                "query": row[2],
                "retrieved_chunks": json.loads(row[3]),
                "answer": row[4],
                "latency_ms": row[5],
            }
        )

    return logs
