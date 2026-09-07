"""
Shared LangChain ChatOllama singleton (SRS Section 7 - Language Model).

`ChatOllama` (from `langchain_ollama`) is loaded once and reused for both
query rewriting and answer generation. All traffic goes through the
LangChain wrapper.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from config import config
from utils.logger import get_logger

logger = get_logger(__name__)

_llm: ChatOllama | None = None


def get_llm() -> ChatOllama:
    """Lazily load and cache the LangChain `ChatOllama` client."""
    global _llm
    if _llm is None:
        logger.info(
            "Loading Ollama chat model: %s @ %s", config.OLLAMA_MODEL, config.OLLAMA_HOST
        )
        _llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_HOST,
            temperature=0.2,
        )
        logger.info("Ollama chat model loaded")
    return _llm
