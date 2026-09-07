"""
Answer generation via LCEL (SRS Section 23 - Answer Generation).

Composes a `ChatPromptTemplate` with the shared `ChatOllama` and a
`StrOutputParser`. The prompt injects the retrieved context (Document
list) plus the original user query and asks the model to answer only
from the supplied context, citing source document/page metadata where
possible.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from generation.ollama_client import get_llm

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided document context. If the answer cannot be found in the "
    "context, say so clearly instead of guessing. When possible, mention "
    "the source page number(s) your answer relies on."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question using only the context above.",
        ),
    ]
)


def format_context(docs: List[Document]) -> str:
    """Format a list of LangChain Documents into a single context string
    for the answer prompt."""
    parts = []
    for d in docs:
        meta = d.metadata or {}
        doc_name = meta.get("document_name", "unknown")
        page = meta.get("page_number", "?")
        parts.append(f"[Source: {doc_name}, page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


_chain: Runnable | None = None


def get_answer_chain() -> Runnable:
    """Return the LCEL chain: prompt -> LLM -> string parser.

    Input payload: `{"question": str, "context": List[Document]}`.
    """
    global _chain
    if _chain is None:
        _chain = _PROMPT | get_llm() | StrOutputParser()
    return _chain
