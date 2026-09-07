"""
Semantic caching (SRS Section 19-22).

LangChain has no built-in semantic cache, so this module provides a
custom `BaseRetriever` (`CacheRetriever`) that wraps the underlying
ChromaDB collection (similarity search over past query embeddings) and
a SQLite table for cache metadata (answer text, document
identifier/version, timestamps).

The legacy function-style API (`lookup`, `store`, `clear_cache`,
`document_signature`) is kept for backwards compatibility with the
existing pipeline call sites.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import config
from embeddings.model import embed_text, get_embeddings
from utils.logger import get_logger

logger = get_logger(__name__)

_sqlite_conn: Optional[sqlite3.Connection] = None
_cache_vectorstore: Chroma | None = None


def _get_cache_vectorstore() -> Chroma:
    global _cache_vectorstore
    if _cache_vectorstore is None:
        os.makedirs(config.CACHE_PATH, exist_ok=True)
        _cache_vectorstore = Chroma(
            collection_name=config.CACHE_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=config.CACHE_PATH,
        )
    return _cache_vectorstore


def _get_sqlite_conn() -> sqlite3.Connection:
    global _sqlite_conn
    if _sqlite_conn is None:
        db_dir = os.path.dirname(config.SQLITE_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        _sqlite_conn = sqlite3.connect(config.SQLITE_PATH)
        _sqlite_conn.row_factory = sqlite3.Row
        _sqlite_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                cache_id TEXT PRIMARY KEY,
                original_query TEXT NOT NULL,
                rewritten_query TEXT,
                answer TEXT NOT NULL,
                document_signature TEXT NOT NULL,
                retrieved_chunk_ids TEXT,
                created_at REAL NOT NULL
            );
            """
        )
        _sqlite_conn.commit()
    return _sqlite_conn


def document_signature(document_path: str) -> str:
    """A signature identifying the current document/version so cache
    entries from a stale document state are not treated as valid
    (SRS Section 22 - Cache Safety)."""
    if not os.path.exists(document_path):
        return hashlib.sha256(document_path.encode("utf-8")).hexdigest()[:16]
    stat = os.stat(document_path)
    raw = f"{document_path}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheHit:
    cache_id: str
    original_query: str
    answer: str
    similarity: float


def lookup(
    query: str,
    doc_signature: str,
    threshold: Optional[float] = None,
) -> Optional[CacheHit]:
    threshold = config.CACHE_SIMILARITY_THRESHOLD if threshold is None else threshold
    vs = _get_cache_vectorstore()

    logger.info("Cache lookup started")

    if vs._collection.count() == 0:
        logger.info("Cache MISS (empty cache)")
        return None

    query_embedding = embed_text(query)
    results = vs._collection.query(
        query_embeddings=[query_embedding],
        n_results=min(5, vs._collection.count()),
        where={"document_signature": doc_signature},
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not ids:
        logger.info("Cache MISS (no entries for current document version)")
        return None

    best_distance = distances[0]
    # ChromaDB default distance is squared L2 on normalized-ish embeddings;
    # convert to an approximate similarity in [0, 1] for threshold comparison.
    best_similarity = 1.0 / (1.0 + best_distance)
    logger.info("Best cache similarity: %.4f", best_similarity)

    if best_similarity >= threshold:
        conn = _get_sqlite_conn()
        row = conn.execute(
            "SELECT * FROM cache_metadata WHERE cache_id = ?", (ids[0],)
        ).fetchone()
        if row is None:
            logger.warning("Cache metadata missing for cache_id=%s; treating as MISS", ids[0])
            return None
        logger.info("Cache HIT")
        return CacheHit(
            cache_id=row["cache_id"],
            original_query=row["original_query"],
            answer=row["answer"],
            similarity=best_similarity,
        )

    logger.info("Cache MISS")
    return None


def store(
    original_query: str,
    rewritten_query: str,
    answer: str,
    doc_signature: str,
    retrieved_chunk_ids: Optional[list] = None,
) -> str:
    vs = _get_cache_vectorstore()
    conn = _get_sqlite_conn()

    cache_id = str(uuid.uuid4())
    embedding = embed_text(original_query)

    vs._collection.add(
        ids=[cache_id],
        embeddings=[embedding],
        documents=[original_query],
        metadatas=[{"document_signature": doc_signature}],
    )

    conn.execute(
        """
        INSERT INTO cache_metadata
            (cache_id, original_query, rewritten_query, answer, document_signature,
             retrieved_chunk_ids, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cache_id,
            original_query,
            rewritten_query,
            answer,
            doc_signature,
            ",".join(retrieved_chunk_ids or []),
            time.time(),
        ),
    )
    conn.commit()

    logger.info("New semantic cache entry stored (cache_id=%s)", cache_id)
    return cache_id


def clear_cache() -> int:
    """Clear all cached entries. Returns the number of entries removed."""
    global _cache_vectorstore
    conn = _get_sqlite_conn()
    count = conn.execute("SELECT COUNT(*) AS n FROM cache_metadata;").fetchone()["n"]
    conn.execute("DELETE FROM cache_metadata;")
    conn.commit()

    if _cache_vectorstore is not None:
        try:
            _cache_vectorstore.delete_collection()
        except Exception:
            pass
        _cache_vectorstore = None

    logger.info("Semantic cache cleared (%d entries removed)", count)
    return count


class CacheRetriever(BaseRetriever):
    """LangChain `BaseRetriever` that wraps the semantic cache.

    The retriever is paired with a `document_signature` (set at
    construction time) so it only matches cache entries for the current
    document version. Each returned `Document` carries the cached
    `answer`, `cache_id`, and `cache_similarity` in its metadata so a
    downstream Runnable can short-circuit the rest of the chain.
    """

    document_signature_value: str
    threshold: float = Field(default=config.CACHE_SIMILARITY_THRESHOLD)
    top_k: int = Field(default=1)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        hit = lookup(query, self.document_signature_value, threshold=self.threshold)
        if hit is None:
            return []
        return [
            Document(
                page_content=hit.original_query,
                metadata={
                    "cache_id": hit.cache_id,
                    "answer": hit.answer,
                    "cache_similarity": hit.similarity,
                    "cache_hit": True,
                },
            )
        ]


def get_cache_retriever(document_signature_value: str) -> CacheRetriever:
    """Convenience constructor for the cache retriever."""
    return CacheRetriever(document_signature_value=document_signature_value)
