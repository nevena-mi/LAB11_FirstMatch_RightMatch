"""Lightweight document chunking helpers.

The module keeps chunking intentionally simple: fixed-size character windows
with overlap, while preserving source traceability from normalized documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .normalization import NormalizedDocument

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100


@dataclass(slots=True)
class Chunk:
    """Retrieval-ready chunk derived from a normalized document."""

    text: str
    chunk_id: str
    metadata: dict
    source_id: str
    document_type: str
    document_id: str


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def _chunk_boundaries(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    if not text:
        return [(0, 0)]

    boundaries: list[tuple[int, int]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        boundaries.append((start, end))
        if end >= text_length:
            break
        start = end - chunk_overlap

    return boundaries


def chunk_document(
    document: NormalizedDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a normalized document into deterministic text chunks."""

    _validate_chunk_settings(chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(_chunk_boundaries(document.text, chunk_size, chunk_overlap)):
        chunk_text = document.text[start:end]
        chunk_metadata = dict(document.metadata)
        chunk_metadata.update(
            {
                "document_id": document.document_id,
                "chunk_index": index,
                "chunk_start_char": start,
                "chunk_end_char": end,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )
        chunk_id = f"{document.document_id}:chunk-{index:04d}"
        chunks.append(
            Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                metadata=chunk_metadata,
                source_id=document.source_id,
                document_type=document.document_type,
                document_id=document.document_id,
            )
        )

    return chunks


def chunk_documents(
    documents: Iterable[NormalizedDocument],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a collection of normalized documents."""

    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return all_chunks
