"""
Shared LangChain embedding model singleton (SRS Section 8 - Embedding Model).

Wraps `HuggingFaceEmbeddings` (langchain-huggingface) around the
`all-MiniLM-L6-v2` SentenceTransformer model. The instance is loaded
exactly once and reused for document chunk embeddings, user query
embeddings, and semantic cache embeddings.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily load and cache the LangChain HuggingFaceEmbeddings wrapper."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded")
    return _embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning plain Python lists of floats."""
    return get_embeddings().embed_documents(texts)


def embed_text(text: str) -> list[float]:
    return get_embeddings().embed_query(text)
