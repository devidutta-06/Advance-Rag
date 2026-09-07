"""
Application-wide logging setup (SRS Section 24 - Logging Requirements).

Uses Python's built-in `logging` module only. Logs to both console (INFO+
by default) and a rotating-free log file (all levels down to DEBUG),
allowing the developer to inspect every major processing stage.
"""

import logging
import os
import sys

from config import config

_CONFIGURED = False


def setup_logging(log_level: str | None = None) -> None:
    """Configure root application logging. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (log_level or config.LOG_LEVEL or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = os.path.dirname(config.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers control actual filtering
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Quiet down noisy third-party libraries unless we're in DEBUG mode.
    if level > logging.DEBUG:
        for noisy in ("chromadb", "httpx", "urllib3", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
