"""Utility helpers for project setup.

This module stays deliberately small so the notebook can validate the
environment before the rest of the RAG pipeline exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "COHERE_API_KEY", "PINECONE_KEY")


def load_environment(dotenv_path: Path | None = None) -> bool:
    """Load environment variables from `.env`.

    Args:
        dotenv_path: Optional explicit path to a dotenv file. When omitted,
            the repo-level `.env` file is used.

    Returns:
        True if a dotenv file was found and loaded, otherwise False.
    """

    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency issue in local env only
        raise RuntimeError(
            "python-dotenv is required to load environment variables from .env."
        ) from exc

    path = dotenv_path or (PROJECT_ROOT / ".env")
    return load_dotenv(dotenv_path=path)


def validate_environment(required_keys: Iterable[str] | None = None) -> list[str]:
    """Return a list of required environment variables that are missing.

    The function only checks presence, never values, so secrets are not exposed.
    """

    keys = tuple(required_keys or DEFAULT_REQUIRED_ENV_VARS)
    return [key for key in keys if not os.getenv(key)]


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path
