"""
Sparse retrieval via SQLite/FTS5 (SRS Section 13 - Sparse Retrieval).

LangChain has no native FTS5 retriever, so this module exposes a custom
`BaseRetriever` subclass (`FTS5Retriever`) that delegates to the
BM25-ranked FTS5 index. The legacy function-style API
(`sparse_search`, `index_chunks`, `clear_sparse_index`) is kept for
backwards compatibility with the rest of the pipeline.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import config
from ingestion.metadata import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)

_connection: Optional[sqlite3.Connection] = None

_FTS5_TOKEN_RE = re.compile(r"[^a-zA-Z0-9\s]")


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        db_dir = os.path.dirname(config.SQLITE_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        _connection = sqlite3.connect(config.SQLITE_PATH)
        _connection.row_factory = sqlite3.Row
        _init_schema(_connection)
        logger.info("SQLite database initialized at %s", config.SQLITE_PATH)
    return _connection


def _init_schema(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS sparse_index USING fts5(
                chunk_id UNINDEXED,
                text,
                document_name UNINDEXED,
                document_path UNINDEXED,
                page_number UNINDEXED,
                tokenize = 'porter'
            );
            """
        )
        conn.commit()
        logger.info("FTS5 index initialized")
    except sqlite3.OperationalError as exc:
        logger.error("FTS5 is not available in this SQLite build: %s", exc)
        raise


def clear_sparse_index() -> None:
    conn = _get_connection()
    conn.execute("DELETE FROM sparse_index;")
    conn.commit()


def index_chunks(chunks: List[Chunk]) -> None:
    if not chunks:
        logger.warning("No chunks to index in SQLite/FTS5")
        return

    conn = _get_connection()
    conn.executemany(
        """
        INSERT INTO sparse_index (chunk_id, text, document_name, document_path, page_number)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (c.chunk_id, c.text, c.document_name, c.document_path, c.page_number)
            for c in chunks
        ],
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) AS n FROM sparse_index;").fetchone()["n"]
    logger.info("Number of sparse documents indexed: %d", count)


def _sanitize_fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH query: strip special chars, OR the terms."""
    cleaned = _FTS5_TOKEN_RE.sub(" ", query)
    terms = [t for t in cleaned.split() if t]
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)


@dataclass
class SparseResult:
    chunk_id: str
    text: str
    metadata: Dict
    score: float  # BM25 rank score (lower = more relevant, per SQLite FTS5)
    rank: int


def sparse_search(query: str, top_k: Optional[int] = None) -> List[SparseResult]:
    """Legacy function-style API. Returns `SparseResult` records."""
    k = top_k or config.SPARSE_TOP_K
    conn = _get_connection()

    total = conn.execute("SELECT COUNT(*) AS n FROM sparse_index;").fetchone()["n"]
    if total == 0:
        logger.warning("Sparse retrieval skipped: sparse index is empty")
        return []

    logger.info("Sparse retrieval started")
    fts_query = _sanitize_fts_query(query)

    try:
        rows = conn.execute(
            """
            SELECT chunk_id, text, document_name, document_path, page_number,
                   bm25(sparse_index) AS score
            FROM sparse_index
            WHERE sparse_index MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, k),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("Sparse query failed (%s); returning no sparse results", exc)
        return []

    results = [
        SparseResult(
            chunk_id=row["chunk_id"],
            text=row["text"],
            metadata={
                "document_name": row["document_name"],
                "document_path": row["document_path"],
                "page_number": row["page_number"],
                "chunk_id": row["chunk_id"],
            },
            score=row["score"],
            rank=i + 1,
        )
        for i, row in enumerate(rows)
    ]
    logger.info("Number of sparse results: %d", len(results))
    return results


class FTS5Retriever(BaseRetriever):
    """LangChain-compatible retriever backed by our SQLite/FTS5 index.

    Use as:
        retriever = FTS5Retriever(top_k=config.SPARSE_TOP_K)
        docs = retriever.invoke("user question")
    """

    top_k: int = Field(default=8)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Ensure schema exists for the first call.
        _ = _get_connection()
        results = sparse_search(query, top_k=self.top_k)
        return [
            Document(
                page_content=r.text,
                metadata={**r.metadata, "bm25_score": r.score, "rank": r.rank},
            )
            for r in results
        ]


def get_sparse_retriever(top_k: Optional[int] = None) -> FTS5Retriever:
    """Return a LangChain `BaseRetriever` for the FTS5 index."""
    return FTS5Retriever(top_k=top_k or config.SPARSE_TOP_K)
