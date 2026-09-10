"""
PDF text extraction (SRS Section 9 - PDF Processing).

Uses LangChain's `PyPDFLoader` to load a PDF into a list of
`langchain_core.documents.Document` objects, one per page. Each Document
preserves the page number via `metadata["page"]` (0-indexed) so it can
be carried through chunking and retrieval.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or contains no usable text."""


@dataclass
class PageText:
    """Lightweight per-page view used by the chunker."""
    page_number: int  # 1-indexed
    text: str


# "layout" mode pads text with extra spaces/blank lines to preserve visual
# column alignment; collapse that down so chunk/sentence-count stats and
# the semantic chunker aren't thrown off by artificial whitespace runs.
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text


def _to_page_text(documents: list[Document]) -> List[PageText]:
    """Convert LangChain Documents (one per page) into PageText records
    with 1-indexed page numbers."""
    pages: List[PageText] = []
    for i, doc in enumerate(documents, start=1):
        raw_text = (doc.page_content or "").strip()
        pages.append(PageText(page_number=i, text=_normalize_whitespace(raw_text)))
    return pages


def load_pdf_pages(pdf_path: str) -> List[PageText]:
    """Load a PDF via `PyPDFLoader` and return per-page text.

    Raises PDFLoadError for missing files, invalid PDFs, or PDFs with no
    extractable text (SRS Section 31 - Error Handling).
    """
    logger.info("PDF received: %s", pdf_path)

    if not os.path.exists(pdf_path):
        raise PDFLoadError(f"PDF not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise PDFLoadError(
            f"Unsupported file type (only .pdf is supported in Version 1): {pdf_path}"
        )

    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as exc:  # pragma: no cover
        raise PDFLoadError(
            "langchain-community is not installed. Run: pip install langchain-community"
        ) from exc

    logger.info("Text extraction started")
    try:
        # extraction_mode="layout" (vs. the default "plain") reconstructs
        # whitespace from the visual position of text on the page, instead
        # of relying on the PDF's content stream to contain explicit space/
        # newline characters between separately-positioned text runs. This
        # is what fixes bulleted/list layouts extracting as one fused run
        # of text (see PDFLoadError note above / SRS bug: missing spaces
        # between list items such as OCR-VLM system names).
        loader = PyPDFLoader(file_path=pdf_path, extraction_mode="layout")
        documents = loader.load()
    except Exception as exc:
        raise PDFLoadError(f"Invalid or corrupted PDF: {pdf_path} ({exc})") from exc

    if not documents:
        raise PDFLoadError(f"PDF has no pages: {pdf_path}")

    logger.info("PDF pages detected: %d", len(documents))
    pages = _to_page_text(documents)

    total_chars = sum(len(p.text) for p in pages)
    if total_chars < 10:
        raise PDFLoadError(
            f"PDF has insufficient extractable text (possibly scanned/image-based): {pdf_path}"
        )

    logger.info("Text extraction completed")
    return pages