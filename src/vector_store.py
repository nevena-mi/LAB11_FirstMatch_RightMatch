"""Minimal Pinecone vector store helpers.

This module keeps Pinecone integration isolated from embedding generation and
later retrieval logic. It stores embedded chunks with traceable metadata and
supports a small similarity query for validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from .embeddings import EmbeddedChunk

DEFAULT_NAMESPACE = "__default__"
DEFAULT_METRIC = "cosine"


class VectorStoreError(RuntimeError):
    """Raised when Pinecone configuration or operations fail."""


@dataclass(slots=True)
class PineconeConfig:
    """Environment-driven Pinecone configuration."""

    api_key: str
    index_name: str
    cloud: str
    region: str


def _read_config(index_name: str | None = None) -> PineconeConfig:
    api_key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY")
    resolved_index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
    cloud = os.getenv("PINECONE_CLOUD")
    region = os.getenv("PINECONE_REGION")

    missing: list[str] = []
    if not api_key:
        missing.append("PINECONE_API_KEY or PINECONE_KEY")
    if not resolved_index_name:
        missing.append("PINECONE_INDEX_NAME")
    if not cloud:
        missing.append("PINECONE_CLOUD")
    if not region:
        missing.append("PINECONE_REGION")

    if missing:
        raise VectorStoreError(
            "Missing Pinecone configuration: " + ", ".join(missing)
        )

    return PineconeConfig(
        api_key=api_key,
        index_name=resolved_index_name,
        cloud=cloud,
        region=region,
    )


def _get_client(index_name: str | None = None) -> tuple[Pinecone, PineconeConfig]:
    config = _read_config(index_name=index_name)
    return Pinecone(api_key=config.api_key), config


def _sanitize_metadata_value(value: Any) -> Any:
    """Remove None values while preserving valid metadata types."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {
            key: sanitized
            for key, item in value.items()
            if (sanitized := _sanitize_metadata_value(item)) is not None
        }
    if isinstance(value, list):
        return [
            sanitized
            for item in value
            if (sanitized := _sanitize_metadata_value(item)) is not None
        ]
    if isinstance(value, tuple):
        return [
            sanitized
            for item in value
            if (sanitized := _sanitize_metadata_value(item)) is not None
        ]
    return value


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a Pinecone-safe copy of metadata without None values."""

    sanitized = _sanitize_metadata_value(metadata)
    if not isinstance(sanitized, dict):
        return {}
    return sanitized


def _as_vector_record(chunk: EmbeddedChunk) -> dict[str, Any]:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "document_id": chunk.document_id,
            "source_id": chunk.source_id,
            "document_type": chunk.document_type,
            "text": chunk.text,
        }
    )
    return {
        "id": chunk.chunk_id,
        "values": [float(value) for value in chunk.embedding],
        "metadata": _sanitize_metadata(metadata),
    }


def create_index_if_needed(
    dimension: int,
    *,
    index_name: str | None = None,
    metric: str = DEFAULT_METRIC,
) -> str:
    """Create the Pinecone index if it does not already exist."""

    pc, config = _get_client(index_name=index_name)
    if not pc.has_index(config.index_name):
        try:
            pc.create_index(
                name=config.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=config.cloud, region=config.region),
            )
        except Exception as exc:  # noqa: BLE001 - keep network/config failures readable
            raise VectorStoreError(f"Failed to create Pinecone index: {exc}") from exc

    return config.index_name


def get_index(index_name: str | None = None) -> Any:
    """Return a Pinecone index handle."""

    pc, config = _get_client(index_name=index_name)
    try:
        return pc.Index(config.index_name)
    except Exception as exc:  # noqa: BLE001 - keep network/config failures readable
        raise VectorStoreError(f"Failed to open Pinecone index: {exc}") from exc


def upsert_chunk(
    chunk: EmbeddedChunk,
    *,
    index_name: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> Any:
    """Upsert one embedded chunk into Pinecone."""

    return upsert_chunks([chunk], index_name=index_name, namespace=namespace)


def upsert_chunks(
    chunks: list[EmbeddedChunk],
    *,
    index_name: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> Any:
    """Upsert a list of embedded chunks into Pinecone."""

    if not chunks:
        return None

    index = get_index(index_name=index_name)
    vectors = [_as_vector_record(chunk) for chunk in chunks]

    try:
        return index.upsert(vectors=vectors, namespace=namespace)
    except Exception as exc:  # noqa: BLE001 - keep network/config failures readable
        raise VectorStoreError(f"Failed to upsert chunks into Pinecone: {exc}") from exc


def query_index(
    query_embedding: list[float],
    top_k: int = 5,
    *,
    index_name: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> Any:
    """Query Pinecone with a dense embedding vector."""

    index = get_index(index_name=index_name)
    try:
        return index.query(
            vector=[float(value) for value in query_embedding],
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )
    except Exception as exc:  # noqa: BLE001 - keep network/config failures readable
        raise VectorStoreError(f"Failed to query Pinecone index: {exc}") from exc
