# Semantic-Cached Advanced RAG Pipeline

A lightweight, CPU-only Advanced RAG system for answering questions about a
PDF document. Built on **LangChain** (LCEL) and **Ollama**, with semantic
chunking, hybrid (dense + sparse) retrieval fused with Reciprocal Rank
Fusion, Maximum Marginal Relevance for diversity, query rewriting, and
semantic caching — all running fully locally.

Built to match the accompanying
`Software Requirements Specification (SRS).md`.

## Features

- **LangChain Expression Language (LCEL)** orchestration of the full RAG
  pipeline (cache branch + retrieval branch composed as `Runnable` chains)
- PDF text extraction (page-aware) via LangChain `PyPDFLoader`
- Semantic chunking via LangChain `SemanticChunker`
  (`langchain_experimental`) — embedding-distance breakpoints
- Chunk metadata (document name/path, page number, chunk id) as
  `langchain_core.documents.Document` objects
- Dense retrieval — `langchain_chroma.Chroma` + `HuggingFaceEmbeddings`
  (`all-MiniLM-L6-v2`)
- Sparse retrieval — custom `BaseRetriever` subclass over SQLite `FTS5`
  (BM25 ranking)
- Hybrid retrieval fused with Reciprocal Rank Fusion (RRF) via
  `langchain_classic.EnsembleRetriever`
- Maximum Marginal Relevance (MMR) — both custom on RRF candidates and
  via `langchain-chroma.max_marginal_relevance_search`
- Query rewriting via a `ChatPromptTemplate | ChatOllama | StrOutputParser`
  LCEL chain (with automatic fallback to the original query)
- Answer generation via an LCEL chain injecting the retrieved context
- Semantic cache — custom `BaseRetriever` subclass over ChromaDB +
  SQLite metadata
- Two-sink logging: compact console (no third-party chatter) + detailed
  file log (always DEBUG)
- Simple interactive CLI — no web UI required

Explicitly **out of scope** for Version 1: cross-encoder reranking, DOCX/
image/web input, RAG agents, cloud vector DBs, Redis, LangSmith, GPU
requirements. See the SRS for the full list.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) installed and running locally
- ~8 GB RAM, CPU-only is sufficient

## Setup

1. Install and start Ollama, then pull the chat model:

   ```bash
   ollama pull llama3.2
   ```

2. Create a virtual environment and install Python dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   The first install will pull in `langchain`, `langchain-core`,
   `langchain-community`, `langchain-text-splitters`, `langchain-classic`,
   `langchain-huggingface`, `langchain-ollama`, `langchain-chroma`, and
   `langchain-experimental` (for `SemanticChunker`).

3. Place a PDF at `data/sample.pdf` (or point `--pdf` / `PDF_PATH` at your
   own file).

## Running

### 1. Ingest the document (run once, or when PDF changes)

```bash
python ingest.py
```

Optional flags:

```bash
python ingest.py --pdf data/my_document.pdf       # Custom PDF path
python ingest.py --reindex                        # Force rebuild indexes
python ingest.py --log-level DEBUG                # Verbose app logging
python ingest.py --quiet                          # Only show warnings/errors
```

### 2. Query the document (interactive)

```bash
python query.py
```

Optional flags:

```bash
python query.py --pdf data/my_document.pdf        # Custom PDF path
python query.py --log-level DEBUG                 # Verbose app logging
python query.py --quiet                           # Only show warnings/errors
```

### CLI commands (in query.py)

| Command         | Description                                        |
|-----------------|-----------------------------------------------------|
| `/help`         | Show available commands                             |
| `/exit`         | Exit the application                                |
| `/clear-cache`  | Delete all semantic cache entries                    |
| `/reindex`      | Reprocess the PDF and rebuild indexes (runs ingestion) |

Anything else typed at the prompt is treated as a natural-language question
about the document.

## Configuration

All tunables are centralized in `config.py` and overridable via environment
variables, e.g.:

```bash
export OLLAMA_MODEL=llama3.2:latest
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export DENSE_TOP_K=8
export SPARSE_TOP_K=8
export MMR_TOP_K=5
export MMR_LAMBDA=0.5
export CACHE_SIMILARITY_THRESHOLD=0.72
export LOG_LEVEL=INFO
export PDF_PATH=data/sample.pdf
```

See `config.py` for the full list (`CHROMA_PATH`, `CACHE_PATH`,
`SQLITE_PATH`, `RRF_K`, `SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE`, etc.).

## Project structure

```text
advanced-rag/
├── ingest.py                 # Data ingestion entry point (run once per PDF)
├── query.py                  # Interactive query entry point
├── pipeline.py               # Query-time orchestration (LCEL Runnable chain)
├── config.py                 # Centralized configuration
├── requirements.txt
├── README.md
│
├── data/                     # Place your PDF(s) here
├── storage/                  # ChromaDB + SQLite + logs (created at runtime)
│
├── embeddings/
│   └── model.py              # Shared HuggingFaceEmbeddings singleton
│                              # (langchain-huggingface)
│
├── ingestion/
│   ├── pdf_loader.py          # PDF text extraction (PyPDFLoader)
│   ├── chunker.py             # Semantic chunking (SemanticChunker)
│   ├── metadata.py            # Chunk metadata → langchain_core.documents.Document
│   └── pipeline.py            # Ingestion orchestration
│
├── retrieval/
│   ├── dense.py               # langchain-chroma vector store + retriever
│   ├── sparse.py               # FTS5Retriever (BaseRetriever over SQLite/FTS5)
│   ├── fusion.py               # EnsembleRetriever (RRF over dense + sparse)
│   └── mmr.py                  # Custom MMR on RRF candidates +
│                                # langchain-chroma max_marginal_relevance_search
│
├── query/
│   └── rewriter.py            # LCEL chain: ChatPromptTemplate | ChatOllama
│                                # | StrOutputParser (with fallback)
│
├── cache/
│   └── semantic_cache.py      # CacheRetriever (BaseRetriever) over ChromaDB
│                                # + SQLite metadata
│
├── generation/
│   ├── ollama_client.py       # Shared ChatOllama singleton (langchain-ollama)
│   └── answer.py               # LCEL chain for answer generation
│
└── utils/
    └── logger.py               # Two-sink logging + stderr filter
```

## How it works

**Ingestion:** PDF → text extraction via `PyPDFLoader` → semantic chunking
via `SemanticChunker` → metadata-attached `Document` objects → indexed in
both ChromaDB (dense) and SQLite/FTS5 (sparse).

**Query:** user query → embedded → semantic cache check (via `CacheRetriever`).
- **Cache hit:** cached answer returned immediately, no LLM/retrieval call.
- **Cache miss:** query rewritten by an LCEL chain
  (`ChatPromptTemplate | ChatOllama | StrOutputParser`, falls back to the
  original query if rewriting fails) → dense (`Chroma.as_retriever()`) and
  sparse (`FTS5Retriever`) retrieval in parallel → results fused with RRF
  (`EnsembleRetriever` / `reciprocal_rank_fusion`) → MMR selects a diverse
  top-k context set → answer generated by an LCEL chain that injects the
  retrieved context → answer stored in the semantic cache.

The full query-time flow is a single `Runnable` chain in
`pipeline.build_rag_chain()` — inspectable via `chain.get_graph()` and
swappable at the component level (e.g. drop in a custom retriever or
add a tracing callback).

## LangChain vs. custom code

Almost every component is now a LangChain primitive or a thin LangChain
wrapper:

| Component | Library |
|-----------|---------|
| PDF loading | `langchain_community.document_loaders.PyPDFLoader` |
| Semantic chunking | `langchain_experimental.text_splitter.SemanticChunker` |
| Embeddings | `langchain_huggingface.HuggingFaceEmbeddings` |
| Vector store | `langchain_chroma.Chroma` |
| LLM | `langchain_ollama.ChatOllama` |
| Dense retriever | `Chroma.as_retriever(search_type="mmr")` |
| Hybrid retriever | `langchain_classic.retrievers.EnsembleRetriever` |
| Custom retrievers | `langchain_core.retrievers.BaseRetriever` (FTS5, Cache) |
| Chain composition | `langchain_core.runnables.{RunnablePassthrough, RunnableBranch, RunnableLambda}` |
| Prompts | `langchain_core.prompts.ChatPromptTemplate` |
| Output parsing | `langchain_core.output_parsers.StrOutputParser` |
| Document type | `langchain_core.documents.Document` |

Two components remain custom because LangChain has no native equivalent:

- **`FTS5Retriever`** — SQLite/FTS5 BM25 sparse retriever (LangChain has
  no FTS5 integration).
- **`CacheRetriever`** — semantic cache over ChromaDB + SQLite metadata
  (LangChain has no built-in semantic cache).

Both implement `langchain_core.retrievers.BaseRetriever` so they plug
seamlessly into the rest of the pipeline.

## Notes on design choices

- **No cross-encoder / LLM reranking:** intentionally excluded to keep
  Version 1 fast on CPU-only, 8 GB RAM hardware. MMR provides a lightweight
  diversity-aware selection instead.
- **Single embedding model:** `all-MiniLM-L6-v2` is loaded once and reused
  for document chunks, queries, and the semantic cache — no second model.
- **Everything local:** no data leaves the machine except for the initial
  one-time package/model downloads.
- **`langchain-experimental` and `langchain-classic`:** both are listed
  in `requirements.txt` and used for `SemanticChunker` and
  `EnsembleRetriever` respectively. They are upstream-supported packages
  even though their status differs from the core library.
