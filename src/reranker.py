"""Dedicated reranking helpers for retrieved chunks.

This module keeps Step 10 separate from baseline retrieval and Step 9
relevance scoring. It applies a deterministic, local scoring strategy that
behaves like a lightweight cross-encoder: the query and chunk text are scored
together, without loading or downloading any external model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .relevance_scoring import ScoredChunk
from .retrieval import RetrievedChunk

DEFAULT_RERANK_TOP_K = 3
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(slots=True)
class RerankedChunk:
    """A chunk with a dedicated reranking score."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    source_id: str
    document_type: str
    retrieval_score: float
    rerank_score: float
    relevance_score: float | None = None
    reasoning: str | None = None
    rank: int = 0


@dataclass(slots=True)
class LocalCrossEncoderReranker:
    """Deterministic reranker that scores query and chunk text together."""

    text_weight: float = 0.8
    retrieval_weight: float = 0.2
    relevance_weight: float = 0.35
    phrase_bonus: float = 0.2

    def score(self, query: str, chunk: RetrievedChunk | ScoredChunk) -> float:
        """Return a reranking score for a single chunk."""

        text_score = _score_text_match(query, chunk.text, phrase_bonus=self.phrase_bonus)
        retrieval_score = _retrieval_score(chunk)
        relevance_score = _relevance_score(chunk)

        if relevance_score is None:
            return min(
                1.0,
                (self.text_weight * text_score) + (self.retrieval_weight * retrieval_score),
            )

        query_signal = (text_score + relevance_score) / 2.0
        return min(
            1.0,
            (self.text_weight * query_signal)
            + (self.retrieval_weight * retrieval_score)
            + (self.relevance_weight * relevance_score),
        )


DEFAULT_RERANKER = LocalCrossEncoderReranker()


def _normalize_tokens(text: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in _STOPWORDS]


def _score_text_match(query: str, text: str, *, phrase_bonus: float) -> float:
    """Score how well the chunk text matches the query text."""

    query_tokens = _normalize_tokens(query)
    text_tokens = _normalize_tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0

    query_set = set(query_tokens)
    text_set = set(text_tokens)
    overlap = query_set & text_set

    coverage = len(overlap) / len(query_set)
    precision = len(overlap) / len(text_set)

    score = (0.6 * coverage) + (0.4 * precision)

    normalized_query = " ".join(query_tokens)
    normalized_text = " ".join(text_tokens)
    if normalized_query and normalized_query in normalized_text:
        score += phrase_bonus

    ordered_hits = 0
    for left, right in zip(query_tokens, query_tokens[1:]):
        if left in text_set and right in text_set:
            ordered_hits += 1
    if len(query_tokens) > 1:
        score += 0.1 * (ordered_hits / (len(query_tokens) - 1))

    return min(1.0, score)


def _retrieval_score(chunk: RetrievedChunk | ScoredChunk) -> float:
    value = getattr(chunk, "retrieval_score", getattr(chunk, "score", 0.0))
    return float(value)


def _relevance_score(chunk: RetrievedChunk | ScoredChunk) -> float | None:
    value = getattr(chunk, "relevance_score", None)
    if value is None:
        return None
    return float(value)


def _reasoning(chunk: RetrievedChunk | ScoredChunk) -> str | None:
    value = getattr(chunk, "reasoning", None)
    if value is None:
        return None
    return str(value)


def _to_reranked_chunk(
    chunk: RetrievedChunk | ScoredChunk,
    *,
    rerank_score: float,
    rank: int,
) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=str(getattr(chunk, "chunk_id", "")),
        document_id=str(getattr(chunk, "document_id", "")),
        text=str(getattr(chunk, "text", "")),
        metadata=dict(getattr(chunk, "metadata", {}) or {}),
        source_id=str(getattr(chunk, "source_id", "")),
        document_type=str(getattr(chunk, "document_type", "")),
        retrieval_score=_retrieval_score(chunk),
        rerank_score=rerank_score,
        relevance_score=_relevance_score(chunk),
        reasoning=_reasoning(chunk),
        rank=rank,
    )


def rerank(
    query: str,
    chunks: list[RetrievedChunk | ScoredChunk],
    top_k: int = DEFAULT_RERANK_TOP_K,
    *,
    reranker: LocalCrossEncoderReranker | None = None,
) -> list[RerankedChunk]:
    """Rerank already retrieved chunks without calling Pinecone."""

    if not chunks or top_k <= 0:
        return []

    scorer = reranker or DEFAULT_RERANKER
    scored_chunks: list[tuple[RetrievedChunk | ScoredChunk, float]] = []
    for chunk in chunks:
        scored_chunks.append((chunk, scorer.score(query, chunk)))

    ranked = sorted(
        scored_chunks,
        key=lambda item: (
            -item[1],
            -_retrieval_score(item[0]),
            -(_relevance_score(item[0]) if _relevance_score(item[0]) is not None else -1.0),
            str(getattr(item[0], "chunk_id", "")),
        ),
    )

    return [
        _to_reranked_chunk(chunk, rerank_score=score, rank=rank)
        for rank, (chunk, score) in enumerate(ranked[:top_k], start=1)
    ]
