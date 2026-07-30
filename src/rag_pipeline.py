"""Thin orchestration for the completed RAG pipeline stages.

This module intentionally coordinates existing components only:
retrieval -> relevance scoring -> reranking -> metadata filtering.
It does not load environments, create clients, or implement any new ranking
or embedding logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .metadata_filtering import filter_chunks
from .relevance_scoring import DEFAULT_RELEVANCE_MODEL, score_relevance
from .reranker import RerankedChunk, rerank
from .retrieval import retrieve_query

DEFAULT_PIPELINE_TOP_K_RETRIEVAL = 10
DEFAULT_PIPELINE_TOP_K_RESULTS = 3


def run_rag_pipeline(
    query: str,
    filters: Mapping[str, Any] | None = None,
    top_k_retrieval: int = DEFAULT_PIPELINE_TOP_K_RETRIEVAL,
    top_k_results: int = DEFAULT_PIPELINE_TOP_K_RESULTS,
    *,
    index_name: str | None = None,
    client: Any | None = None,
    relevance_model: str = DEFAULT_RELEVANCE_MODEL,
    reranker: Any | None = None,
) -> list[RerankedChunk]:
    """Run the full retrieval-to-filtering pipeline for a query.

    The returned chunks are the reranked chunks that survive optional metadata
    filtering. The function preserves the order established by reranking.
    """

    retrieved_chunks = retrieve_query(
        query,
        top_k=top_k_retrieval,
        index_name=index_name,
        client=client,
    )
    scored_chunks = score_relevance(
        query,
        retrieved_chunks,
        model=relevance_model,
        client=client,
    )
    reranked_chunks = rerank(
        query,
        scored_chunks,
        top_k=top_k_results,
        reranker=reranker,
    )
    if filters:
        return filter_chunks(reranked_chunks, filters)
    return reranked_chunks


__all__ = [
    "DEFAULT_PIPELINE_TOP_K_RETRIEVAL",
    "DEFAULT_PIPELINE_TOP_K_RESULTS",
    "run_rag_pipeline",
]

