"""
Dense retrieval via langchain-chroma (SRS Section 12 - Dense Retrieval).

Wraps ChromaDB in a `langchain_chroma.Chroma` vector store so we can use
`as_retriever()` and the `max_marginal_relevance_search` helper from
langchain-chroma for MMR. Embeddings are produced by the shared
`HuggingFaceEmbeddings` instance (see `embeddings.model`).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import config
from embeddings.model import get_embeddings
from ingestion.metadata import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)

_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        os.makedirs(config.CHROMA_PATH, exist_ok=True)
        _vectorstore = Chroma(
            collection_name=config.DOCUMENT_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=config.CHROMA_PATH,
        )
        logger.info("ChromaDB initialized at %s", config.CHROMA_PATH)
    return _vectorstore


def get_retriever(top_k: Optional[int] = None):
    """Return a LangChain `VectorStoreRetriever` over the document
    collection with `k = top_k` (default `config.DENSE_TOP_K`)."""
    k = top_k or config.DENSE_TOP_K
    return _get_vectorstore().as_retriever(search_kwargs={"k": k})


def get_mmr_retriever(top_k: Optional[int] = None, fetch_k: int = 20, lambda_mult: Optional[float] = None):
    """Return a retriever that uses `max_marginal_relevance_search`
    (langchain-chroma) for MMR-aware selection."""
    k = top_k or config.MMR_TOP_K
    lam = lambda_mult if lambda_mult is not None else config.MMR_LAMBDA
    return _get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lam},
    )


@dataclass
class DenseResult:
    chunk_id: str
    text: str
    metadata: Dict
    distance: float
    rank: int


def index_chunks(chunks: List[Chunk]) -> None:
    if not chunks:
        logger.warning("No chunks to index in ChromaDB")
        return

    logger.info("Embedding generation started")
    documents = [c.to_document() for c in chunks]
    # Identifiers in ChromaDB must be string-compatible.
    ids = [c.chunk_id for c in chunks]

    _get_vectorstore().add_documents(documents=documents, ids=ids)
    logger.info("Documents indexed")
    logger.info(
        "Number of vectors stored: %d", _get_vectorstore()._collection.count()
    )


def clear_document_collection() -> None:
    """Drop the persistent collection directory and reset the cached
    vectorstore. We can't selectively delete from a single collection in
    a multi-tenant store, so clearing the directory is the cleanest reset
    for a single-tenant setup."""
    global _vectorstore
    if _vectorstore is not None:
        try:
            _vectorstore.delete_collection()
        except Exception:
            pass
        _vectorstore = None
    # Belt-and-braces: also wipe the directory in case of stale on-disk state.
    if os.path.isdir(config.CHROMA_PATH):
        try:
            shutil.rmtree(config.CHROMA_PATH, ignore_errors=True)
        except Exception:
            pass


def clear_document(document_path: str) -> None:
    """Delete all vectors belonging to one document, without wiping the
    whole collection.

    Chroma's `collection.add()` silently skips re-adding an id that
    already exists — it does NOT overwrite it. That means re-running
    ingestion on the same PDF without `--reindex` leaves every existing
    chunk's embedding untouched (e.g. still reflecting pre-fix, fused
    extraction text), even though the new chunk objects were computed
    correctly. Call this before indexing so re-ingesting the same file is
    always safe, independent of the `--reindex` flag.
    """
    vs = _get_vectorstore()
    try:
        vs._collection.delete(where={"document_path": document_path})
        logger.info("Cleared existing dense vectors for document: %s", document_path)
    except Exception as exc:
        logger.warning(
            "Could not clear existing dense vectors for %s: %s", document_path, exc
        )


def dense_search(query: str, top_k: Optional[int] = None) -> List[DenseResult]:
    k = top_k or config.DENSE_TOP_K
    vs = _get_vectorstore()
    if vs._collection.count() == 0:
        logger.warning("Dense retrieval skipped: document collection is empty")
        return []

    logger.info("Dense retrieval started")
    # Use `similarity_search_with_score` (returns cosine distance-like
    # values). `similarity_search_with_relevance_scores` requires scores
    # in [0, 1] and throws on negative values when embeddings are
    # normalized, so it is not safe for our setup.
    docs_and_scores = vs.similarity_search_with_score(
        query, k=min(k, vs._collection.count())
    )

    results: List[DenseResult] = []
    for i, (doc, distance) in enumerate(docs_and_scores, start=1):
        results.append(
            DenseResult(
                chunk_id=doc.metadata.get("chunk_id", f"unknown_{i}"),
                text=doc.page_content,
                metadata=doc.metadata,
                distance=float(distance),
                rank=i,
            )
        )
    logger.info("Number of dense results: %d", len(results))
    return results


def get_vectorstore() -> Chroma:
    """Public accessor for callers that need the underlying langchain-chroma
    vector store (e.g. MMR, raw similarity search)."""
    return _get_vectorstore()