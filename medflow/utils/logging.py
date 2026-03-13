"""Structured logging setup using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_dir: str | None = None, level: str = "INFO") -> None:
    """Configure loguru for structured console + file logging.

    Args:
        log_dir: If provided, also writes logs to this directory.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()  # Remove default handler

    # Console handler — clean format for dev
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File handler — full format for debugging
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            Path(log_dir) / "medflow_{time}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            rotation="50 MB",
            retention="7 days",
        )

    logger.info(f"Logger initialized at level={level}")


__all__ = ["setup_logger", "logger"]
