"""
Chunk metadata generation (SRS Section 11 - Metadata).

Every chunk is associated with at least: document_name, document_path,
page_number, and a unique chunk_id. Output is a list of
`langchain_core.documents.Document` objects so they can be passed
directly to `langchain_chroma.Chroma.from_documents()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from langchain_core.documents import Document

from ingestion.chunker import RawChunk
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Lightweight chunk handle kept for backwards compatibility with the
    ingestion/retrieval code paths. The `text` and `metadata` fields
    match a `langchain_core.documents.Document` 1-to-1."""
    chunk_id: str
    text: str
    document_name: str
    document_path: str
    page_number: int
    metadata: Dict

    def to_document(self) -> Document:
        return Document(page_content=self.text, metadata=dict(self.metadata))


def build_chunks(raw_chunks: List[RawChunk], document_path: str) -> List[Chunk]:
    document_name = os.path.basename(document_path)
    per_page_counter: Dict[int, int] = {}

    chunks: List[Chunk] = []
    for raw in raw_chunks:
        page = raw.page_number
        per_page_counter[page] = per_page_counter.get(page, 0) + 1
        seq = per_page_counter[page]

        stem = os.path.splitext(document_name)[0]
        chunk_id = f"{stem}_p{page}_c{seq:02d}"

        metadata = {
            "document_name": document_name,
            "document_path": document_path,
            "page_number": page,
            "chunk_id": chunk_id,
        }

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=raw.text,
                document_name=document_name,
                document_path=document_path,
                page_number=page,
                metadata=metadata,
            )
        )

    logger.debug("Generated metadata for %d chunks", len(chunks))
    return chunks
