"""
Shared low-level Ollama client helper used by both query rewriting and
answer generation (SRS Section 7 - Language Model Requirements).
"""

from __future__ import annotations

from typing import Optional

from config import config
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaUnavailableError(Exception):
    """Raised when Ollama or the configured model cannot be reached/used."""


def chat(prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
    """Send a single-turn chat request to the local Ollama server."""
    try:
        import ollama
    except ImportError as exc:
        raise OllamaUnavailableError(
            "The 'ollama' Python package is not installed. Run: pip install ollama"
        ) from exc

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        raise OllamaUnavailableError(
            f"Ollama call failed (host={config.OLLAMA_HOST}, model={config.OLLAMA_MODEL}): {exc}"
        ) from exc
