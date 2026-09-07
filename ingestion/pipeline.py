"""
Document ingestion pipeline orchestrator (SRS Section 5.1 - Document
Ingestion). Wires together PDF loading -> semantic chunking -> metadata
generation -> dense (ChromaDB) + sparse (SQLite/FTS5) indexing.
"""

from __future__ import annotations

from ingestion.chunker import chunk_pages
from ingestion.metadata import build_chunks
from ingestion.pdf_loader import load_pdf_pages
from retrieval import dense, sparse
from utils.logger import get_logger

logger = get_logger(__name__)


def ingest_document(pdf_path: str, reindex: bool = False) -> int:
    """Run the full ingestion pipeline for a PDF. Returns the number of
    chunks indexed."""
    if reindex:
        logger.info("Reindex requested: clearing existing dense and sparse indexes")
        dense.clear_document_collection()
        sparse.clear_sparse_index()

    pages = load_pdf_pages(pdf_path)
    raw_chunks = chunk_pages(pages)
    chunks = build_chunks(raw_chunks, document_path=pdf_path)

    dense.index_chunks(chunks)
    sparse.index_chunks(chunks)

    return len(chunks)
