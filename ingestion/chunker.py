"""
Semantic chunking (SRS Section 10 - Semantic Chunking).

Rather than splitting on fixed character/token windows, text is first split
into sentences, each sentence is embedded, and "breakpoints" are inserted
wherever the semantic distance between consecutive sentences exceeds a
configurable percentile threshold. Sentences between breakpoints are
grouped into a single chunk, keeping semantically related content together.

This mirrors the approach used by LangChain's experimental SemanticChunker,
implemented directly here (using the same embedding model already loaded
for the rest of the pipeline) to avoid an extra heavyweight dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import config
from embeddings.model import embed_texts
from ingestion.pdf_loader import PageText
from utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class RawChunk:
    text: str
    page_number: int
    sentence_count: int = field(default=0)


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
    similarity = float(np.dot(a, b) / denom)
    return 1.0 - similarity


def _semantic_split(sentences: List[str]) -> List[List[str]]:
    """Group sentences into semantically coherent groups using embedding
    distance breakpoints, respecting MIN_CHUNK_SENTENCES and MAX_CHUNK_CHARS.
    """
    if len(sentences) <= 1:
        return [sentences] if sentences else []

    vectors = np.array(embed_texts(sentences))
    distances = [
        _cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
    ]

    if not distances:
        return [sentences]

    threshold = float(
        np.percentile(distances, config.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE)
    )

    groups: List[List[str]] = []
    current: List[str] = [sentences[0]]
    current_chars = len(sentences[0])

    for i, dist in enumerate(distances):
        next_sentence = sentences[i + 1]
        is_breakpoint = dist > threshold
        would_overflow = current_chars + len(next_sentence) > config.MAX_CHUNK_CHARS
        enough_sentences = len(current) >= config.MIN_CHUNK_SENTENCES

        if (is_breakpoint or would_overflow) and enough_sentences:
            groups.append(current)
            current = [next_sentence]
            current_chars = len(next_sentence)
        else:
            current.append(next_sentence)
            current_chars += len(next_sentence)

    if current:
        groups.append(current)

    return groups


def chunk_pages(pages: List[PageText]) -> List[RawChunk]:
    """Semantically chunk every page's text, preserving page association."""
    logger.info("Chunking started")

    chunks: List[RawChunk] = []
    for page in pages:
        if not page.text:
            continue
        sentences = _split_sentences(page.text)
        if not sentences:
            continue
        groups = _semantic_split(sentences)
        for group in groups:
            chunk_text = " ".join(group).strip()
            if chunk_text:
                chunks.append(
                    RawChunk(text=chunk_text, page_number=page.page_number, sentence_count=len(group))
                )

    logger.info("Number of chunks generated: %d", len(chunks))
    logger.info("Chunking completed")
    return chunks
