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
import sys

from config import config

_CONFIGURED = False

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
    "filelock",
    "asyncio",
)


def setup_logging(log_level: str | None = None, quiet: bool = False) -> None:
    """Configure root application logging. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

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

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
