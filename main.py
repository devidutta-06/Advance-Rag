#!/usr/bin/env python
"""
Semantic-Cached Advanced RAG Pipeline — Main Entry Point (Deprecated)

This script is kept for backward compatibility. Please use the dedicated entry points:

    python ingest.py    # Ingest a PDF document (run once, or when document changes)
    python query.py     # Interactive question-answering (requires pre-ingested document)

See README.md for details.
"""

from __future__ import annotations

import sys

BANNER = """
====================================
 Advanced RAG
====================================
""".rstrip()

HELP_TEXT = """
This is the legacy entry point. Please use:

  python ingest.py    # Data preparation: load PDF, chunk, embed, index
  python query.py     # Interactive Q&A: ask questions about indexed document

Legacy commands (still work but deprecated):
  --pdf PATH          Specify PDF path (default: data/sample.pdf)
  --log-level LEVEL   DEBUG, INFO, WARNING, ERROR

For help on new entry points:
  python ingest.py --help
  python query.py --help
""".strip()


def main() -> int:
    print(BANNER)
    print(HELP_TEXT)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())