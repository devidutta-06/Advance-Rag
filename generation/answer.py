"""
Answer generation via Ollama (SRS Section 23 - Answer Generation).

The final prompt contains the original user query plus the retrieved
context. The model is instructed to answer only from the supplied context
and to cite source document/page metadata where possible.
"""

from __future__ import annotations

from typing import List

from generation.ollama_client import OllamaUnavailableError, chat
from retrieval.fusion import FusedResult
from utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided document context. If the answer cannot be found in the "
    "context, say so clearly instead of guessing. When possible, mention "
    "the source page number(s) your answer relies on."
)


def _format_context(chunks: List[FusedResult]) -> str:
    parts = []
    for c in chunks:
        doc = c.metadata.get("document_name", "unknown")
        page = c.metadata.get("page_number", "?")
        parts.append(f"[Source: {doc}, page {page}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(original_query: str, context_chunks: List[FusedResult]) -> str:
    logger.info("Ollama generation started")

    if not context_chunks:
        logger.warning("No context available for answer generation")
        answer = (
            "I couldn't find relevant information in the document to answer that question."
        )
        logger.info("Ollama generation completed")
        return answer

    context_text = _format_context(context_chunks)
    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {original_query}\n\n"
        "Answer the question using only the context above."
    )

    try:
        answer = chat(prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.2)
    except OllamaUnavailableError as exc:
        logger.error("Answer generation failed: %s", exc)
        answer = (
            "Answer generation is currently unavailable because Ollama could not be "
            "reached. Please ensure Ollama is running and the configured model is "
            "installed, then try again."
        )

    logger.info("Ollama generation completed")
    return answer
