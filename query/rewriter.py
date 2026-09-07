"""
Query rewriting via Ollama (SRS Section 15 - Query Rewriting).

The original user query is always preserved. The rewritten query is used
for retrieval, but is never a permanent replacement for the original.
If rewriting fails, the system falls back to the original query rather
than terminating the RAG process (SRS NFR-06 - Reliability).
"""

from __future__ import annotations

from dataclasses import dataclass

from generation.ollama_client import OllamaUnavailableError, chat
from utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You rewrite short, ambiguous, or poorly formulated user questions into "
    "clear, specific, standalone questions suitable for searching a document. "
    "Preserve the original intent. Respond with ONLY the rewritten question, "
    "no explanation, no quotes."
)


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    used_fallback: bool


def rewrite_query(original_query: str) -> RewriteResult:
    logger.info("Query rewriting started")
    logger.debug("Original query: %s", original_query)

    try:
        rewritten = chat(prompt=original_query, system=_SYSTEM_PROMPT, temperature=0.2)
        rewritten = rewritten.strip().strip('"').strip()
        if not rewritten:
            raise OllamaUnavailableError("Empty rewrite response")
        logger.info("Rewritten query: %s", rewritten)
        return RewriteResult(original_query=original_query, rewritten_query=rewritten, used_fallback=False)
    except OllamaUnavailableError as exc:
        logger.warning("Query rewriting failed, falling back to original query: %s", exc)
        return RewriteResult(original_query=original_query, rewritten_query=original_query, used_fallback=True)
