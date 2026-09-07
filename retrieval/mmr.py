"""
Maximum Marginal Relevance (SRS Section 17 - Maximum Marginal Relevance).

Implemented in two flavors:

  1. `apply_mmr(query, candidates, ...)` — operates on a list of
     `FusedResult` candidates from the RRF fusion step and re-ranks them
     using cosine similarity. Useful when fusion is the primary source of
     candidates.

  2. `mmr_retriever()` — returns a LangChain `BaseRetriever` that uses
     `langchain-chroma.Chroma.max_marginal_relevance_search` directly.
     Useful for LCEL-style composition.

No separate reranking model (e.g. cross-encoder) is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import config
from embeddings.model import embed_text, embed_texts
from retrieval.dense import get_vectorstore
from retrieval.fusion import FusedResult
from utils.logger import get_logger

logger = get_logger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
    return float(np.dot(a, b) / denom)


def apply_mmr(
    query: str,
    candidates: List[FusedResult],
    top_k: Optional[int] = None,
    lambda_param: Optional[float] = None,
) -> List[FusedResult]:
    top_k = top_k or config.MMR_TOP_K
    lambda_param = config.MMR_LAMBDA if lambda_param is None else lambda_param

    logger.info("MMR started")
    logger.info("Number of candidates: %d", len(candidates))

    if not candidates:
        logger.info("Number of selected chunks: 0")
        return []

    if len(candidates) <= top_k:
        logger.info("Number of selected chunks: %d", len(candidates))
        return candidates

    query_vec = np.array(embed_text(query))
    candidate_vecs = np.array(embed_texts([c.text for c in candidates]))

    relevance = np.array(
        [_cosine_similarity(query_vec, vec) for vec in candidate_vecs]
    )

    selected_idx: List[int] = []
    remaining_idx = list(range(len(candidates)))

    first = int(np.argmax(relevance))
    selected_idx.append(first)
    remaining_idx.remove(first)

    while remaining_idx and len(selected_idx) < top_k:
        best_score = None
        best_i = None
        for i in remaining_idx:
            max_sim_to_selected = max(
                _cosine_similarity(candidate_vecs[i], candidate_vecs[j])
                for j in selected_idx
            )
            mmr_score = (
                lambda_param * relevance[i]
                - (1 - lambda_param) * max_sim_to_selected
            )
            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_i = i
        selected_idx.append(best_i)
        remaining_idx.remove(best_i)

    selected = [candidates[i] for i in selected_idx]

    logger.debug("MMR-selected chunk_ids: %s", [c.chunk_id for c in selected])
    logger.info("Number of selected chunks: %d", len(selected))
    return selected


def mmr_search(
    query: str,
    top_k: Optional[int] = None,
    fetch_k: int = 20,
    lambda_mult: Optional[float] = None,
) -> List[Document]:
    """Run MMR directly against the langchain-chroma vector store.

    Returns LangChain `Document` objects so they can be fed into a
    downstream chain (e.g. the answer-generation prompt)."""
    k = top_k or config.MMR_TOP_K
    lam = lambda_mult if lambda_mult is not None else config.MMR_LAMBDA
    vs = get_vectorstore()
    logger.info("MMR (chroma) started")
    docs: List[Document] = vs.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lam
    )
    logger.info("Number of selected chunks: %d", len(docs))
    return docs


def mmr_retriever(
    top_k: Optional[int] = None,
    fetch_k: int = 20,
    lambda_mult: Optional[float] = None,
) -> BaseRetriever:
    """Return a LangChain `BaseRetriever` that performs MMR via
    langchain-chroma. Useful for LCEL chains."""
    k = top_k or config.MMR_TOP_K
    lam = lambda_mult if lambda_mult is not None else config.MMR_LAMBDA
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lam},
    )
