"""Embedding helpers for chunk objects.

This module keeps embedding generation isolated from retrieval and storage.
It converts `Chunk` objects into embedded records while preserving the
original traceability fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chunking import Chunk

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails for a recoverable reason."""


@dataclass(slots=True)
class EmbeddedChunk:
    """Chunk enriched with an embedding vector."""

    chunk_id: str
    document_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    source_id: str
    document_type: str


def _get_openai_client(client: Any | None = None) -> Any:
    """Return a provided client or create a new OpenAI client lazily."""

    if client is not None:
        return client

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency issue in local env only
        raise EmbeddingError(
            "The openai package is required to generate embeddings."
        ) from exc

    return OpenAI()


def _embed_texts(
    texts: list[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: Any | None = None,
) -> list[list[float]]:
    """Embed a list of texts and return vectors in the same order."""

    openai_client = _get_openai_client(client)
    try:
        response = openai_client.embeddings.create(
            model=model,
            input=texts,
        )
    except Exception as exc:  # noqa: BLE001 - wrap API/auth/network failures cleanly
        raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

    vectors: list[list[float]] = []
    for item in response.data:
        vectors.append([float(value) for value in item.embedding])
    return vectors


def embed_chunk(
    chunk: Chunk,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: Any | None = None,
) -> EmbeddedChunk:
    """Embed a single chunk."""

    embedding = _embed_texts([chunk.text], model=model, client=client)[0]
    return EmbeddedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=embedding,
        metadata=dict(chunk.metadata),
        source_id=chunk.source_id,
        document_type=chunk.document_type,
    )


def embed_chunks(
    chunks: list[Chunk],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: Any | None = None,
) -> list[EmbeddedChunk]:
    """Embed multiple chunks while preserving their original ordering."""

    if not chunks:
        return []

    embeddings = _embed_texts([chunk.text for chunk in chunks], model=model, client=client)
    embedded_chunks: list[EmbeddedChunk] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                embedding=embedding,
                metadata=dict(chunk.metadata),
                source_id=chunk.source_id,
                document_type=chunk.document_type,
            )
        )
    return embedded_chunks
