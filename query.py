#!/usr/bin/env python
"""
Interactive query entry point for the Semantic-Cached Advanced RAG Pipeline.

This script provides the CLI for asking questions about an already-ingested document.
The document must be ingested first using `python ingest.py`.

Usage:
    python query.py                    # Uses config.PDF_PATH
    python query.py --pdf data/my.pdf  # Specify custom PDF (must be pre-ingested)
    python query.py --log-level DEBUG  # Verbose app logging
    python query.py --quiet            # Only show warnings/errors
"""

from __future__ import annotations

import argparse
import sys

from cache import semantic_cache
from config import config
from pipeline import answer_question
from utils.logger import get_logger, setup_logging

HELP_TEXT = """
Available commands:
  /help          Show this help message
  /exit          Exit the application
  /clear-cache   Delete all semantic cache entries
  /reindex       Reprocess the configured PDF and rebuild local indexes

Type a question in natural language to ask about the loaded document.
""".strip()

BANNER = """
====================================
 Advanced RAG — Query Mode
====================================
""".rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the RAG system (document must be pre-ingested)"
    )
    parser.add_argument(
        "--pdf",
        default=config.PDF_PATH,
        help=f"Path to the PDF (must be pre-ingested, default: {config.PDF_PATH})"
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
    logger = get_logger("query")

    logger.info("Query mode started")
    logger.info("Configuration loaded")
    logger.info("PDF path: %s", args.pdf)

    print(BANNER)
    print(f"\nDocument: {args.pdf}")
    print("Note: Document must be pre-ingested. Run 'python ingest.py' first if needed.\n")
    print(HELP_TEXT)
    print()

    document_ready = True  # Assume pre-ingested; /reindex will refresh

    while True:
        try:
            user_input = input("Question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == "/help":
            print("\n" + HELP_TEXT + "\n")
            continue

        if user_input == "/exit":
            print("Goodbye.")
            break

        if user_input == "/clear-cache":
            try:
                removed = semantic_cache.clear_cache()
                print(f"\nCleared {removed} cache entr{'y' if removed == 1 else 'ies'}.\n")
            except Exception as exc:
                logger.error("Failed to clear cache: %s", exc, exc_info=True)
                print(f"\nFailed to clear cache: {exc}\n")
            continue

        if user_input == "/reindex":
            print("\nReindexing document... (this will run ingestion)")
            # Import here to avoid circular dependency
            from ingestion.pipeline import ingest_document
            from ingestion.pdf_loader import PDFLoadError
            try:
                num_chunks = ingest_document(args.pdf, reindex=True)
                document_ready = True
                print(f"Reindex complete. ({num_chunks} chunks indexed)\n")
            except PDFLoadError as exc:
                logger.error("PDF load error during reindex: %s", exc)
                print(f"\nCould not reindex: {exc}\n")
            except Exception as exc:
                logger.error("Unexpected error during reindex: %s", exc, exc_info=True)
                print(f"\nAn unexpected error occurred during reindex: {exc}\n")
            continue

        if user_input.startswith("/"):
            print(f"\nUnknown command: {user_input}\nType /help to see available commands.\n")
            continue

        if not document_ready:
            print("\nNo document is currently indexed. Run 'python ingest.py' first.\n")
            continue

        try:
            outcome = answer_question(user_input, args.pdf)
        except Exception as exc:
            logger.error("Unexpected error while answering question: %s", exc, exc_info=True)
            print(f"\nSomething went wrong while answering your question: {exc}\n")
            continue

        print()
        if outcome.cache_hit:
            print(f"Answer (from cache, similarity={outcome.cache_similarity:.2f}):")
        else:
            print("Answer:")
        print(outcome.answer)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())