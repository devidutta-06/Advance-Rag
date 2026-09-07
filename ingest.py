#!/usr/bin/env python
"""
Data ingestion entry point for the Semantic-Cached Advanced RAG Pipeline.

This script handles document loading, chunking, embedding, and indexing.
Run once per document, or when the document changes.

Usage:
    python ingest.py                    # Uses config.PDF_PATH
    python ingest.py --pdf data/my.pdf  # Specify custom PDF
    python ingest.py --reindex          # Force rebuild indexes
    python ingest.py --log-level DEBUG  # Verbose app logging
    python ingest.py --quiet            # Only show warnings/errors
"""

from __future__ import annotations

import argparse
import sys

from config import config
from ingestion.pdf_loader import PDFLoadError
from ingestion.pipeline import ingest_document
from utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF document into the RAG system (embeddings + indexes)"
    )
    parser.add_argument(
        "--pdf",
        default=config.PDF_PATH,
        help=f"Path to the PDF to ingest (default: {config.PDF_PATH})"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild: clear existing indexes before ingesting"
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        help="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logs (only show warnings/errors on console)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level, quiet=args.quiet)
    logger = get_logger("ingest")

    logger.info("Ingestion started")
    logger.info("Configuration loaded")
    logger.info("PDF path: %s", args.pdf)
    logger.info("Reindex mode: %s", args.reindex)

    print(f"\nPDF: {args.pdf}")
    if args.reindex:
        print("Reindex mode: clearing existing indexes...")
    print("Loading document...")

    try:
        num_chunks = ingest_document(args.pdf, reindex=args.reindex)
        print(f"\nIngestion complete. ({num_chunks} chunks indexed)\n")
        logger.info("Ingestion completed successfully: %d chunks", num_chunks)
        return 0
    except PDFLoadError as exc:
        logger.error("PDF load error: %s", exc)
        print(f"\nCould not load PDF: {exc}\n")
        return 1
    except Exception as exc:
        logger.error("Unexpected error during ingestion: %s", exc, exc_info=True)
        print(f"\nAn unexpected error occurred: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())