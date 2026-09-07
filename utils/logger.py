"""
Application-wide logging setup (SRS Section 24 - Logging Requirements).

Uses Python's built-in `logging` module only. Two output sinks:

  * Console: compact, app-only. Third-party library chatter (httpx, httpcore,
    urllib3, chromadb telemetry, posthog, etc.) is suppressed at the source so
    it never reaches the user's terminal.
  * File:   detailed, full timestamps + logger names. Always captures DEBUG
    so developers can diagnose issues.

Per-module loggers are obtained via `get_logger(name)`. Console level is
controlled by `--log-level` (default INFO) and `--quiet` (suppresses INFO).
File level is always DEBUG.
"""

from __future__ import annotations

import logging
import os
import re
import sys

from config import config

_CONFIGURED = False

# Patterns whose matches on stderr are silently dropped. Some libraries
# (notably `huggingface_hub`) print one-off messages directly to stderr
# instead of going through `logging`, so we have to filter them here.
_STDERR_FILTERS = (
    re.compile(r"unauthenticated requests to the HF Hub"),
    re.compile(r"Please set a HF_TOKEN to enable higher rate limits"),
)


class _FilteredStderr:
    """Wraps `sys.stderr` and drops lines matching the noisiest
    third-party deprecation / auth warnings. Other writes pass through
    unchanged so real errors still surface."""

    def __init__(self, real) -> None:
        self._real = real

    def write(self, msg: str) -> int:
        if not msg:
            return 0
        if any(p.search(msg) for p in _STDERR_FILTERS):
            return len(msg)
        return self._real.write(msg)

    def flush(self) -> None:
        if hasattr(self._real, "flush"):
            self._real.flush()

    def isatty(self) -> bool:
        return getattr(self._real, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._real.fileno()


def _install_stderr_filter() -> None:
    """Install the stderr filter exactly once."""
    if getattr(sys.stderr, "_advanced_rag_filtered", False):
        return
    sys.stderr = _FilteredStderr(sys.stderr)
    sys.stderr._advanced_rag_filtered = True  # type: ignore[attr-defined]

# Loggers that should never reach the console regardless of verbosity.
# These are libraries whose DEBUG/INFO chatter (HTTP request lines, telemetry
# heartbeats, deprecation warnings) drowns out the application's own messages.
_QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "chromadb",
    "chromadb.telemetry",
    "posthog",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "filelock",
    "asyncio",
)


def setup_logging(log_level: str | None = None, quiet: bool = False) -> None:
    """Configure root application logging. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Always filter known-noisy stderr lines (HF Hub auth warnings,
    # etc.) regardless of verbosity.
    _install_stderr_filter()

    level_name = (log_level or config.LOG_LEVEL or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if quiet:
        # --quiet: only show WARNING and above on the console
        console_level = logging.WARNING
    else:
        console_level = level

    log_dir = os.path.dirname(config.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    # File handler always sees DEBUG, so we set root low.
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Compact console format: short time, level, logger (abbrev), message.
    # Wide terminal widths make long logger names wrap awkwardly; truncate
    # to keep each line on one row.
    console_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_fmt)
    root.addHandler(console_handler)

    # Detailed file format: full timestamp, padded level + logger for greppability.
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers at the source so they never reach
    # either handler. WARNING level is enough to surface real problems
    # (e.g., connection failures) without spamming every HTTP call.
    for noisy in _QUIET_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
        # Disable propagation of their own loggers so even if a child logs
        # something, it stays at WARNING.
        logging.getLogger(noisy).propagate = True  # still let our handlers filter

    # The HF Hub "unauthenticated requests" warning is a one-shot
    # informational notice (not a real problem on local-only setups).
    # It logs at WARNING, so our _QUIET_LOGGERS rule isn't enough; we
    # also have to filter the message itself.
    class _HFHubFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return "unauthenticated" not in msg and "HF_TOKEN" not in msg

    for noisy in ("huggingface_hub", "huggingface_hub.utils._http"):
        logging.getLogger(noisy).addFilter(_HFHubFilter())

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
