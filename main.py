"""
CLI entry point for the Semantic-Cached Advanced RAG Pipeline
(SRS Section 26 - CLI Requirements, Section 27 - CLI Commands).

Usage:
    python main.py
    python main.py --pdf data/sample.pdf
    python main.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import os
import sys

from cache import semantic_cache
from config import config
from ingestion.pdf_loader import PDFLoadError
from ingestion.pipeline import ingest_document
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
 Advanced RAG
====================================
""".rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic-Cached Advanced RAG Pipeline")
    parser.add_argument("--pdf", default=config.PDF_PATH, help="Path to the PDF to load")
    parser.add_argument("--log-level", default=config.LOG_LEVEL, help="DEBUG, INFO, WARNING, ERROR")
    return parser.parse_args()


def load_document(pdf_path: str, logger) -> bool:
    """Attempt to ingest the configured document. Returns True on success."""
    print(f"\nPDF: {pdf_path}\n")
    print("Loading document...")
    try:
        num_chunks = ingest_document(pdf_path)
        print("Building index...")
        print(f"\nReady. ({num_chunks} chunks indexed)\n")
        return True
    except PDFLoadError as exc:
        logger.error("PDF load error: %s", exc)
        print(f"\nCould not load PDF: {exc}\n")
        return False
    except Exception as exc:  # Embedding model / Chroma / SQLite init failures, etc.
        logger.error("Unexpected error during ingestion: %s", exc, exc_info=True)
        print(f"\nAn unexpected error occurred while loading the document: {exc}\n")
        return False


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    logger = get_logger("main")

    logger.info("Application started")
    logger.info("Configuration loaded")

    print(BANNER)

    document_ready = load_document(args.pdf, logger)

    print(HELP_TEXT)
    print()

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
            print("\nReindexing document...")
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
            print("\nNo document is currently indexed. Try /reindex once a valid PDF is available.\n")
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
