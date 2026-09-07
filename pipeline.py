"""
Query-time pipeline orchestrator (SRS Section 5.2 - Query Processing).

Wires together: semantic cache lookup -> (on miss) query rewriting ->
hybrid dense + sparse retrieval -> RRF fusion -> MMR -> answer generation ->
semantic cache storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from cache import semantic_cache
from config import config
from generation.answer import generate_answer
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
    selected_chunks: Optional[List[FusedResult]] = None
    cache_similarity: Optional[float] = None


def answer_question(query: str, pdf_path: str) -> QueryOutcome:
    logger.info("Original query: %s", query)

    doc_signature = semantic_cache.document_signature(pdf_path)

    cache_hit = semantic_cache.lookup(query, doc_signature)
    if cache_hit is not None:
        return QueryOutcome(
            answer=cache_hit.answer,
            cache_hit=True,
            original_query=query,
            cache_similarity=cache_hit.similarity,
        )

    rewrite_result = rewrite_query(query)
    effective_query = rewrite_result.rewritten_query

    dense_results = dense.dense_search(effective_query)
    sparse_results = sparse.sparse_search(effective_query)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    selected = apply_mmr(effective_query, fused)

    answer = generate_answer(query, selected)

    semantic_cache.store(
        original_query=query,
        rewritten_query=effective_query,
        answer=answer,
        doc_signature=doc_signature,
        retrieved_chunk_ids=[c.chunk_id for c in selected],
    )

    return QueryOutcome(
        answer=answer,
        cache_hit=False,
        original_query=query,
        rewritten_query=effective_query,
        selected_chunks=selected,
    )
