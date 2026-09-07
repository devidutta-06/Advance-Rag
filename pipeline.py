"""
Query-time pipeline orchestrator (SRS Section 5.2 - Query Processing).

Wires together: semantic cache lookup -> (on miss) query rewriting ->
hybrid dense + sparse retrieval -> RRF fusion -> MMR -> answer
generation -> semantic cache storage.

The orchestration is expressed as a single LCEL `Runnable` chain
(`build_rag_chain`) plus a thin imperative `answer_question` entry
point for the existing CLI. All subcomponents are LangChain
`BaseRetriever` / `Runnable` instances so the chain can be inspected
via `.get_graph()` or extended (e.g. add tracing, swap retrievers).
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from cache import semantic_cache
from config import config
from generation.answer import format_context, get_answer_chain
from query.rewriter import rewrite_query
from retrieval import dense, sparse
from retrieval.fusion import FusedResult, reciprocal_rank_fusion
from retrieval.mmr import apply_mmr
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryOutcome:
    answer: str
    cache_hit: bool
    original_query: str
    rewritten_query: Optional[str] = None
    cache_similarity: Optional[float] = None


def _candidates_for_query(effective_query: str) -> List[FusedResult]:
    """Run dense + sparse retrieval and fuse with RRF."""
    dense_results = dense.dense_search(effective_query)
    sparse_results = sparse.sparse_search(effective_query)
    return reciprocal_rank_fusion(dense_results, sparse_results)


def _select_chunks(query: str, fused: List[FusedResult]) -> List[Document]:
    """Apply MMR and convert FusedResult -> langchain Document for
    downstream prompt injection."""
    selected = apply_mmr(query, fused)
    return [
        Document(page_content=c.text, metadata=c.metadata) for c in selected
    ]


# Module-level handler functions for LCEL `RunnableLambda` — defining
# these at module scope (rather than as inner functions) avoids
# Pydantic schema-generation issues with nested `Dict[str, Any]`
# annotations.
def _on_miss(payload: Dict[str, Any], doc_sig: str) -> Dict[str, Any]:
    original_query = payload["original_query"]
    rewrite = rewrite_query(original_query)
    effective = rewrite.rewritten_query
    fused = _candidates_for_query(effective)
    chunks = _select_chunks(effective, fused)
    context_text = format_context(chunks)
    try:
        answer = get_answer_chain().invoke(
            {"question": original_query, "context": context_text}
        )
    except Exception as exc:
        logger.error("LCEL answer chain failed: %s", exc)
        answer = (
            "Answer generation is currently unavailable because Ollama could "
            "not be reached. Please ensure Ollama is running and the configured "
            "model is installed, then try again."
        )
    # Persist in the cache.
    semantic_cache.store(
        original_query=original_query,
        rewritten_query=effective,
        answer=answer,
        doc_signature=doc_sig,
        retrieved_chunk_ids=[c.metadata.get("chunk_id", "") for c in chunks],
    )
    return {
        "original_query": original_query,
        "rewritten_query": effective,
        "answer": answer,
        "cache_hit": False,
        "cache_similarity": None,
    }


def _on_hit(payload: Dict[str, Any], doc_sig: str) -> Dict[str, Any]:
    # Re-use the `CacheHit` stashed by `_is_cache_hit` so we don't query
    # the cache twice. Falls back to a fresh lookup if the stash is
    # missing (e.g. when the branch was bypassed).
    hit = payload.get("_cache_hit_obj")
    if hit is None:
        hit = semantic_cache.lookup(payload["original_query"], doc_sig)
    if hit is None:
        # Race: cache was cleared between the branch check and now.
        # Fall through to the miss branch.
        return _on_miss(payload, doc_sig)
    return {
        "original_query": payload["original_query"],
        "rewritten_query": None,
        "answer": hit.answer,
        "cache_hit": True,
        "cache_similarity": hit.similarity,
    }


def _is_cache_hit(payload: Dict[str, Any], doc_sig: str) -> bool:
    hit = semantic_cache.lookup(payload["original_query"], doc_sig)
    # Stash the hit so the on_hit branch can read it without re-querying.
    payload["_cache_hit_obj"] = hit
    return hit is not None


def build_rag_chain(pdf_path: str) -> Runnable:
    """Compose the full LCEL RAG chain for a specific document.

    Input: `{"original_query": "..."}`.
    Output: dict with `answer`, `cache_hit`, `original_query`,
    `rewritten_query`, `cache_similarity`.

    Implementation note: we bind `doc_sig` to the lambdas via
    `functools.partial` so the LCEL graph schema generator can
    introspect each runnable's input/output without tripping over
    Pydantic's handling of forward references in `Dict[str, Any]`.
    """
    doc_sig = semantic_cache.document_signature(pdf_path)

    return (
        RunnablePassthrough()
        | RunnableBranch(
            (
                RunnableLambda(partial(_is_cache_hit, doc_sig=doc_sig)).with_config(
                    run_name="is_cache_hit"
                ),
                RunnableLambda(partial(_on_hit, doc_sig=doc_sig)).with_config(
                    run_name="on_cache_hit"
                ),
            ),
            RunnableLambda(partial(_on_miss, doc_sig=doc_sig)).with_config(
                run_name="on_cache_miss"
            ),
        )
    )


def answer_question(query: str, pdf_path: str) -> QueryOutcome:
    """Imperative entry point used by the CLI. Internally runs the LCEL
    chain and projects the result back into the legacy dataclass."""
    logger.info("Original query: %s", query)
    chain = build_rag_chain(pdf_path)
    result = chain.invoke({"original_query": query})
    return QueryOutcome(
        answer=result["answer"],
        cache_hit=result["cache_hit"],
        original_query=result["original_query"],
        rewritten_query=result.get("rewritten_query"),
        cache_similarity=result.get("cache_similarity"),
    )
