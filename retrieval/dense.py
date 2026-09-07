"""
Dense retrieval via ChromaDB (SRS Section 12 - Dense Retrieval).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import config
from embeddings.model import embed_text, embed_texts
from ingestion.metadata import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        import chromadb

        os.makedirs(config.CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        logger.info("ChromaDB initialized at %s", config.CHROMA_PATH)
    return _client


def get_document_collection():
    client = _get_client()
    return client.get_or_create_collection(name=config.DOCUMENT_COLLECTION)


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

    collection = get_document_collection()
    logger.info("Embedding generation started")
    embeddings = embed_texts([c.text for c in chunks])
    logger.info("Number of embeddings generated: %d", len(embeddings))

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    logger.info("Documents indexed")
    logger.info("Number of vectors stored: %d", collection.count())


def clear_document_collection() -> None:
    client = _get_client()
    try:
        client.delete_collection(config.DOCUMENT_COLLECTION)
    except Exception:
        pass


def dense_search(query: str, top_k: Optional[int] = None) -> List[DenseResult]:
    top_k = top_k or config.DENSE_TOP_K
    collection = get_document_collection()

    if collection.count() == 0:
        logger.warning("Dense retrieval skipped: document collection is empty")
        return []

    logger.info("Dense retrieval started")
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    dense_results = [
        DenseResult(chunk_id=cid, text=doc, metadata=meta, distance=dist, rank=i + 1)
        for i, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances))
    ]
    logger.info("Number of dense results: %d", len(dense_results))
    return dense_results
