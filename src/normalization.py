"""Common document normalization helpers.

This module keeps the shared document layer intentionally small:
it converts PDF pages and transcription results into one stable shape
without adding chunking or retrieval logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pdf_processor import ExtractedPDFPage
from .transcription import TranscriptionResult


@dataclass(slots=True)
class NormalizedDocument:
    """Shared document representation used by later pipeline stages."""

    document_id: str
    text: str
    metadata: dict[str, Any]
    source_id: str
    document_type: str


def _normalized_document_id(*parts: str) -> str:
    """Build a stable, deterministic identifier from source fields."""

    return ":".join(part for part in parts if part)


def normalize_pdf_page(page: ExtractedPDFPage) -> NormalizedDocument:
    """Normalize an extracted PDF page into the shared document schema."""

    document_id = _normalized_document_id(
        page.source_id,
        page.document_type,
        page.filename,
        f"page-{page.page_number}",
    )
    return NormalizedDocument(
        document_id=document_id,
        text=page.text,
        metadata=page.metadata,
        source_id=page.source_id,
        document_type=page.document_type,
    )


def normalize_transcription(result: TranscriptionResult) -> NormalizedDocument:
    """Normalize a transcription result into the shared document schema."""

    source_id = str(result.metadata.get("source", "podcast"))
    document_type = "podcast_transcript"
    filename = str(result.metadata.get("filename", "transcript"))
    document_id = _normalized_document_id(source_id, document_type, filename)
    return NormalizedDocument(
        document_id=document_id,
        text=result.text,
        metadata=result.metadata,
        source_id=source_id,
        document_type=document_type,
    )
