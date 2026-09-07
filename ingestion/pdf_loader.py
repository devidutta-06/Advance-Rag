"""
PDF text extraction (SRS Section 9 - PDF Processing).

Extracts text page-by-page so page numbers can be preserved and later
associated with chunk metadata. Uses pypdf, which is lightweight, pure
Python, and compatible with LangChain-style document loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or contains no usable text."""


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str


def load_pdf_pages(pdf_path: str) -> List[PageText]:
    """Load a PDF and return per-page extracted text.

    Raises PDFLoadError for missing files, invalid PDFs, or PDFs with no
    extractable text (SRS Section 31 - Error Handling).
    """
    logger.info("PDF received: %s", pdf_path)

    if not os.path.exists(pdf_path):
        raise PDFLoadError(f"PDF not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise PDFLoadError(f"Unsupported file type (only .pdf is supported in Version 1): {pdf_path}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise PDFLoadError("pypdf is not installed. Run: pip install pypdf") from exc

    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        raise PDFLoadError(f"Invalid or corrupted PDF: {pdf_path} ({exc})") from exc

    num_pages = len(reader.pages)
    logger.info("PDF pages detected: %d", num_pages)

    if num_pages == 0:
        raise PDFLoadError(f"PDF has no pages: {pdf_path}")

    logger.info("Text extraction started")
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract text from page %d: %s", i, exc)
            text = ""
        pages.append(PageText(page_number=i, text=text.strip()))

    total_chars = sum(len(p.text) for p in pages)
    if total_chars < 10:
        raise PDFLoadError(
            f"PDF has insufficient extractable text (possibly scanned/image-based): {pdf_path}"
        )

    logger.info("Text extraction completed")
    return pages
