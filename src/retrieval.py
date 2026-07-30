"""Baseline retrieval helpers built on top of the Pinecone vector store.

This module stays small on purpose: it embeds a query, calls the existing
vector store wrapper, and returns retrieval-ready records with similarity
scores and traceability fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embeddings import DEFAULT_EMBEDDING_MODEL, _get_openai_client
from .vector_store import query_index

DEFAULT_RETRIEVAL_TOP_K = 5


@dataclass(slots=True)
class RetrievedChunk:
    """A Pinecone match returned by baseline retrieval."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    source_id: str
    document_type: str
    score: float


def _query_embedding(
    query_text: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: Any | None = None,
) -> list[float]:
    """Embed a query using the same OpenAI model as chunk embeddings."""

    openai_client = _get_openai_client(client)
    try:
        response = openai_client.embeddings.create(
            model=model,
            input=[query_text],
        )
    except Exception as exc:  # noqa: BLE001 - wrap API/auth/network failures cleanly
        raise RuntimeError(f"Failed to embed query text: {exc}") from exc

    return [float(value) for value in response.data[0].embedding]


def _to_retrieved_chunk(match: Any) -> RetrievedChunk:
    metadata = dict(getattr(match, "metadata", {}) or {})
    return RetrievedChunk(
        chunk_id=str(getattr(match, "id", "")),
        document_id=str(metadata.get("document_id", "")),
        text=str(metadata.get("text", "")),
        metadata=metadata,
        source_id=str(metadata.get("source_id", "")),
        document_type=str(metadata.get("document_type", "")),
        score=float(getattr(match, "score", 0.0)),
    )


def retrieve(
    query_embedding: list[float],
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    *,
    index_name: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top matches for a dense query embedding."""

    response = query_index(query_embedding, top_k=top_k, index_name=index_name)
    matches = getattr(response, "matches", []) or []
    return [_to_retrieved_chunk(match) for match in matches]


def retrieve_query(
    query_text: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    *,
    index_name: str | None = None,
    client: Any | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[RetrievedChunk]:
    """Embed a query and retrieve the most relevant Pinecone matches."""

    query_embedding = _query_embedding(query_text, model=model, client=client)
    return retrieve(query_embedding, top_k=top_k, index_name=index_name)
