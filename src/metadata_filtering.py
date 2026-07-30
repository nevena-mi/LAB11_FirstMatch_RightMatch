"""Exact-match metadata filtering for retrieved or reranked chunks.

This module intentionally stays small and local-only. It filters chunk-like
objects by comparing requested filter values against top-level traceability
fields first and then against the chunk metadata dictionary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

ChunkT = TypeVar("ChunkT")

_MISSING = object()

_TOP_LEVEL_FIELDS = {
    "chunk_id",
    "document_id",
    "document_type",
    "metadata",
    "rank",
    "reasoning",
    "relevance_score",
    "rerank_score",
    "retrieval_score",
    "score",
    "source_id",
    "text",
}


def _get_candidate_value(chunk: Any, key: str) -> Any:
    """Return the value for a filter key from the chunk or its metadata."""

    if key in _TOP_LEVEL_FIELDS and hasattr(chunk, key):
        return getattr(chunk, key)

    metadata = getattr(chunk, "metadata", None)
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]

    if hasattr(chunk, key):
        return getattr(chunk, key)

    return _MISSING


def _matches_filters(chunk: Any, filters: Mapping[str, Any]) -> bool:
    """Return ``True`` when all filter conditions match exactly."""

    for key, expected in filters.items():
        candidate = _get_candidate_value(chunk, key)
        if candidate is _MISSING or candidate != expected:
            return False
    return True


def filter_chunks(
    chunks: Iterable[ChunkT],
    filters: Mapping[str, Any] | None,
) -> list[ChunkT]:
    """Return chunks whose traceability fields and metadata match ``filters``.

    Filtering is exact-match and AND-combined across every provided key.
    The original chunk objects are returned unchanged.
    """

    chunk_list = list(chunks)
    if not filters:
        return chunk_list

    return [chunk for chunk in chunk_list if _matches_filters(chunk, filters)]

