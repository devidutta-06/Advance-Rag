"""
Reciprocal Rank Fusion (SRS Section 16 - Reciprocal Rank Fusion).

Combines dense and sparse retrieval rankings into a single unified ranking,
preserving chunk_id, dense rank, sparse rank, fusion score, and source
retrieval method for downstream MMR selection and debug logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import config
from retrieval.dense import DenseResult
from retrieval.sparse import SparseResult
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
