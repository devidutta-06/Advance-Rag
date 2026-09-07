# Semantic-Cached Advanced RAG Pipeline

A lightweight, CPU-only Advanced RAG system for answering questions about a
PDF document. Implements semantic chunking, hybrid (dense + sparse)
retrieval fused with Reciprocal Rank Fusion, Maximum Marginal Relevance for
diversity, query rewriting, and semantic caching — all running fully
locally with [Ollama](https://ollama.com) as the LLM runtime.

Built to match the accompanying `Software_Requirements_Specification__SRS_.md`.

## Features

- PDF text extraction (page-aware)
- Semantic chunking (embedding-distance breakpoints, not fixed windows)
- Chunk metadata (document name/path, page number, chunk id)
- Dense retrieval — ChromaDB + `all-MiniLM-L6-v2`
- Sparse retrieval — SQLite `FTS5` (BM25 ranking)
- Hybrid retrieval fused with Reciprocal Rank Fusion (RRF)
- Maximum Marginal Relevance (MMR) for diverse, non-redundant context
- Query rewriting via Ollama, with automatic fallback to the original query
- Semantic caching (ChromaDB + SQLite) to avoid repeated LLM calls
- Detailed stage-by-stage logging (console + file)
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

3. Place a PDF at `data/sample.pdf` (or point `--pdf` / `PDF_PATH` at your
   own file).

## Running

```bash
python main.py
```

Optional flags:

```bash
python main.py --pdf data/my_document.pdf --log-level DEBUG
```

### CLI commands

| Command         | Description                                        |
|-----------------|-----------------------------------------------------|
| `/help`         | Show available commands                             |
| `/exit`         | Exit the application                                |
| `/clear-cache`  | Delete all semantic cache entries                    |
| `/reindex`      | Reprocess the configured PDF and rebuild the indexes |

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

See `config.py` for the full list (`CHROMA_PATH`, `SQLITE_PATH`,
`CACHE_PATH`, `RRF_K`, etc.).

## Project structure

```text
advanced-rag/
├── main.py                  # CLI entry point
├── pipeline.py               # Query-time orchestration (cache → rewrite → retrieve → fuse → MMR → generate)
├── config.py                 # Centralized configuration
├── requirements.txt
├── README.md
│
├── data/                     # Place your PDF(s) here
├── storage/                  # ChromaDB + SQLite + logs (created at runtime)
│
├── embeddings/
│   └── model.py               # Shared all-MiniLM-L6-v2 singleton
│
├── ingestion/
│   ├── pdf_loader.py           # PDF text extraction
│   ├── chunker.py              # Semantic chunking
│   ├── metadata.py             # Chunk metadata construction
│   └── pipeline.py             # Ingestion orchestration
│
├── retrieval/
│   ├── dense.py                # ChromaDB dense retrieval
│   ├── sparse.py                # SQLite/FTS5 sparse retrieval
│   ├── fusion.py                 # Reciprocal Rank Fusion
│   └── mmr.py                    # Maximum Marginal Relevance
│
├── query/
│   └── rewriter.py             # Ollama-based query rewriting (with fallback)
│
├── cache/
│   └── semantic_cache.py       # Semantic cache (ChromaDB + SQLite metadata)
│
├── generation/
│   ├── ollama_client.py        # Shared Ollama chat helper
│   └── answer.py                # Final answer generation
│
└── utils/
    └── logger.py                # Centralized logging setup
```

## How it works

**Ingestion:** PDF → text extraction (page-aware) → semantic chunking →
metadata generation → indexed in both ChromaDB (dense) and SQLite/FTS5
(sparse).

**Query:** user query → embedded → semantic cache check.
- **Cache hit:** cached answer returned immediately, no LLM/retrieval call.
- **Cache miss:** query rewritten by Ollama (falls back to the original
  query if rewriting fails) → dense + sparse retrieval run in parallel →
  results fused with RRF → MMR selects a diverse top-k context set →
  Ollama generates the final answer from that context → answer stored in
  the semantic cache.

## Notes on design choices

- **No cross-encoder / LLM reranking:** intentionally excluded to keep
  Version 1 fast on CPU-only, 8 GB RAM hardware. MMR provides a lightweight
  diversity-aware selection instead.
- **Single embedding model:** `all-MiniLM-L6-v2` is loaded once and reused
  for document chunks, queries, and the semantic cache — no second model.
- **Everything local:** no data leaves the machine except for the initial
  one-time package/model downloads.



# > Improvements
1. main.py contain both data preparation and RAG so the DB is re indexing again and again. Separate the data preparation (embeding and storing ) which need to run once or when new pdf needed
2. 
3. LOG is showing very deatiled -Https - all .. reduce it makeit more compact yet effective same for terminal 