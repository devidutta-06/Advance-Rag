"""
Centralized configuration for the Semantic-Cached Advanced RAG Pipeline.

All configurable values live here and can be overridden via environment
variables so the rest of the codebase never hard-codes paths, model names,
or tunable parameters (per SRS Section 28 - Configuration).
"""

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- Language model (Ollama) ---
    OLLAMA_MODEL: str = field(default_factory=lambda: _env_str("OLLAMA_MODEL", "llama3.2:latest"))
    OLLAMA_HOST: str = field(default_factory=lambda: _env_str("OLLAMA_HOST", "http://localhost:11434"))

    # --- Embedding model ---
    EMBEDDING_MODEL: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # --- Storage paths ---
    BASE_DATA_DIR: str = field(default_factory=lambda: _env_str("BASE_DATA_DIR", "data"))
    CHROMA_PATH: str = field(default_factory=lambda: _env_str("CHROMA_PATH", "storage/chroma"))
    SQLITE_PATH: str = field(default_factory=lambda: _env_str("SQLITE_PATH", "storage/rag.db"))
    CACHE_PATH: str = field(default_factory=lambda: _env_str("CACHE_PATH", "storage/chroma"))

    # --- Chroma collection names ---
    DOCUMENT_COLLECTION: str = field(default_factory=lambda: _env_str("DOCUMENT_COLLECTION", "document_chunks"))
    CACHE_COLLECTION: str = field(default_factory=lambda: _env_str("CACHE_COLLECTION", "semantic_cache"))

    # --- PDF ---
    PDF_PATH: str = field(default_factory=lambda: _env_str("PDF_PATH", r"C:\Users\DeviduttaParida\Desktop\advanced-rag\data\Research Report.pdf"))

    # --- Chunking ---
    SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE: float = field(
        default_factory=lambda: _env_float("SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE", 85.0)
    )
    MIN_CHUNK_SENTENCES: int = field(default_factory=lambda: _env_int("MIN_CHUNK_SENTENCES", 1))
    MAX_CHUNK_CHARS: int = field(default_factory=lambda: _env_int("MAX_CHUNK_CHARS", 1800))

    # --- Retrieval ---
    DENSE_TOP_K: int = field(default_factory=lambda: _env_int("DENSE_TOP_K", 8))
    SPARSE_TOP_K: int = field(default_factory=lambda: _env_int("SPARSE_TOP_K", 8))
    RRF_K: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    # --- MMR ---
    MMR_TOP_K: int = field(default_factory=lambda: _env_int("MMR_TOP_K", 5))
    MMR_LAMBDA: float = field(default_factory=lambda: _env_float("MMR_LAMBDA", 0.5))

    # --- Semantic cache ---
    CACHE_SIMILARITY_THRESHOLD: float = field(
        default_factory=lambda: _env_float("CACHE_SIMILARITY_THRESHOLD", 0.72)
    )

    # --- Logging ---
    LOG_LEVEL: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))
    LOG_FILE: str = field(default_factory=lambda: _env_str("LOG_FILE", "storage/app.log"))


config = Config()
