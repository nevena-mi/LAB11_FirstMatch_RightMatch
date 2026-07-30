"""Compatibility re-export for the dedicated reranker module."""

from __future__ import annotations

from .reranker import DEFAULT_RERANKER, LocalCrossEncoderReranker, RerankedChunk, rerank

__all__ = [
    "DEFAULT_RERANKER",
    "LocalCrossEncoderReranker",
    "RerankedChunk",
    "rerank",
]
