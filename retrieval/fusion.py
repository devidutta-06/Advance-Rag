"""
Hybrid retrieval + Reciprocal Rank Fusion (SRS Section 14-16).

Combines dense and sparse retrieval rankings into a single unified
ranking, preserving chunk_id, dense rank, sparse rank, fusion score, and
source retrieval method for downstream MMR selection and debug logging.

The LangChain `EnsembleRetriever` (from `langchain.retrievers`) implements
RRF over multiple retrievers. We use it for the LCEL-style flow and also
expose the original function-style API (`reciprocal_rank_fusion`) for
backwards compatibility with the existing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import config
from retrieval.dense import DenseResult, get_retriever
from retrieval.sparse import SparseResult, get_sparse_retriever
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FusedResult:
    chunk_id: str
    text: str
    metadata: Dict
    fusion_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    sources: List[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    dense_results: List[DenseResult],
    sparse_results: List[SparseResult],
    k: Optional[int] = None,
) -> List[FusedResult]:
    k = k or config.RRF_K
    logger.info("RRF fusion started")

    combined: Dict[str, FusedResult] = {}

    for d in dense_results:
        score = 1.0 / (k + d.rank)
        combined[d.chunk_id] = FusedResult(
            chunk_id=d.chunk_id,
            text=d.text,
            metadata=d.metadata,
            fusion_score=score,
            dense_rank=d.rank,
            sources=["dense"],
        )

    for s in sparse_results:
        score = 1.0 / (k + s.rank)
        if s.chunk_id in combined:
            existing = combined[s.chunk_id]
            existing.fusion_score += score
            existing.sparse_rank = s.rank
            existing.sources.append("sparse")
        else:
            combined[s.chunk_id] = FusedResult(
                chunk_id=s.chunk_id,
                text=s.text,
                metadata=s.metadata,
                fusion_score=score,
                sparse_rank=s.rank,
                sources=["sparse"],
            )

    fused = sorted(combined.values(), key=lambda r: r.fusion_score, reverse=True)

    for i, r in enumerate(fused):
        logger.debug(
            "RRF candidate #%d chunk_id=%s dense_rank=%s sparse_rank=%s score=%.6f sources=%s",
            i + 1, r.chunk_id, r.dense_rank, r.sparse_rank, r.fusion_score, r.sources,
        )

    logger.info("Number of combined candidates: %d", len(fused))
    return fused


def get_hybrid_retriever(
    top_k_dense: Optional[int] = None,
    top_k_sparse: Optional[int] = None,
) -> BaseRetriever:
    """Return a LangChain `EnsembleRetriever` that performs RRF over
    the dense (Chroma) and sparse (FTS5) retrievers.

    The ensemble uses `RRF_K` as the RRF constant and the per-retriever
    `top_k` values from `config` (or the explicit overrides)."""
    from langchain_classic.retrievers import EnsembleRetriever

    dense_retriever = get_retriever(top_k=top_k_dense)
    sparse_retriever = get_sparse_retriever(top_k=top_k_sparse)
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[1.0, 1.0],
        c=config.RRF_K,
    )
