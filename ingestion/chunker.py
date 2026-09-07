"""
Semantic chunking (SRS Section 10 - Semantic Chunking).

Uses LangChain's experimental `SemanticChunker` to split each page's text
into semantically coherent chunks. The chunker embeds consecutive sentences
and inserts breakpoints where the cosine distance exceeds a configurable
percentile threshold. This produces `langchain_core.documents.Document`
objects (one chunk per Document) with the same `page_number` metadata the
retrieval layer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from config import config
from embeddings.model import get_embeddings
from ingestion.pdf_loader import PageText
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RawChunk:
    """Intermediate chunk record produced by chunking and consumed by
    `ingestion.metadata.build_chunks`."""
    text: str
    page_number: int
    sentence_count: int = field(default=0)


_chunker: SemanticChunker | None = None


def _get_chunker() -> SemanticChunker:
    """Lazily build the SemanticChunker. Shares the same embedding model
    already loaded for the rest of the pipeline (no second model)."""
    global _chunker
    if _chunker is None:
        _chunker = SemanticChunker(
            embeddings=get_embeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=config.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE,
            min_chunk_size=config.MIN_CHUNK_SENTENCES,
        )
    return _chunker


def _approx_sentence_count(text: str) -> int:
    """Cheap sentence counter for the metadata field. The chunker's own
    sentence splitting is internal; we just need a count for logging."""
    count = 0
    for ch in text:
        if ch in ".!?":
            count += 1
    return max(count, 1)


def chunk_pages(pages: List[PageText]) -> List[RawChunk]:
    """Semantically chunk every page's text, preserving page association.

    Returns `RawChunk` records (one per chunk) carrying the page number
    and an approximate sentence count.
    """
    logger.info("Chunking started")
    chunker = _get_chunker()

    chunks: List[RawChunk] = []
    for page in pages:
        if not page.text:
            continue
        page_doc = Document(page_content=page.text, metadata={"page_number": page.page_number})
        try:
            split_docs = chunker.split_documents([page_doc])
        except Exception as exc:  # pragma: no cover
            logger.warning("Semantic chunking failed on page %d: %s; falling back to single chunk", page.page_number, exc)
            split_docs = [page_doc]

        for d in split_docs:
            text = (d.page_content or "").strip()
            if not text:
                continue
            chunks.append(
                RawChunk(
                    text=text,
                    page_number=page.page_number,
                    sentence_count=_approx_sentence_count(text),
                )
            )

    logger.info("Number of chunks generated: %d", len(chunks))
    logger.info("Chunking completed")
    return chunks
