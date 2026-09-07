"""
Shared embedding model singleton (SRS Section 8 - Embedding Model).

`all-MiniLM-L6-v2` is loaded exactly once during application startup and
reused for document chunk embeddings, user query embeddings, and semantic
cache embeddings. No second embedding model is introduced.
"""

from __future__ import annotations

from typing import List

from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

_model = None


def get_embedding_model():
    """Lazily load and cache the SentenceTransformer embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts, returning plain Python lists of floats."""
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]
