"""
utils/logger.py
================
Centralised logging configuration for the entire application.
Writes to both console (colored) and a rotating log file.
"""

import logging
import logging.handlers
from pathlib import Path

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "app.log"
LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a named logger with console + file handlers.

    Args:
        name:  Logger name (typically __name__ of calling module).
        level: Logging level (default INFO).

    Returns:
        logging.Logger
    """
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Avoid duplicate handlers on re-import

    formatter = logging.Formatter(fmt=LOG_FMT, datefmt=DATE_FMT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # Rotating file handler – max 5 MB per file, keep last 3
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
