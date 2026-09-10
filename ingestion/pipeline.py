"""
Document ingestion pipeline orchestrator (SRS Section 5.1 - Document
Ingestion). Wires together PDF loading -> semantic chunking -> metadata
generation -> dense (ChromaDB) + sparse (SQLite/FTS5) indexing. All
subcomponents are now LangChain-based (`PyPDFLoader`,
`SemanticChunker`, `langchain_core.documents.Document`,
`langchain_chroma.Chroma`).
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
    else:
        # Always clear any existing chunks for THIS document before
        # re-adding, even when --reindex wasn't passed. Without this,
        # re-running ingest.py on the same file silently corrupts both
        # indexes: Chroma skips re-adding an existing chunk_id (leaving
        # stale embeddings live), and FTS5 has no uniqueness constraint
        # at all (so sparse rows just accumulate as duplicates). Both
        # were observed in production logs before this fix.
        logger.info("Clearing existing indexed chunks for this document before re-ingesting")
        dense.clear_document(pdf_path)
        sparse.clear_document(pdf_path)

    pages = load_pdf_pages(pdf_path)
    raw_chunks = chunk_pages(pages)
    chunks = build_chunks(raw_chunks, document_path=pdf_path)

    dense.index_chunks(chunks)
    sparse.index_chunks(chunks)

    return len(chunks)