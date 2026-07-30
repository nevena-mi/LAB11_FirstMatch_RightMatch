"""PDF extraction helpers for the RAG lab.

The module intentionally stays small: extract one record per page with enough
metadata to preserve source traceability for later grounding and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PDFProcessingError(RuntimeError):
    """Raised when PDF extraction fails for a recoverable reason."""


@dataclass(slots=True)
class ExtractedPDFPage:
    """Single page of extracted PDF content."""

    text: str
    filename: str
    source_id: str
    document_type: str
    page_number: int
    metadata: dict[str, Any]


def _import_pdf_reader():
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency issue in local env only
        raise PDFProcessingError(
            "pypdf is required for PDF extraction. Install project dependencies "
            "and rerun the notebook."
        ) from exc

    return PdfReader


def extract_pdf_pages(
    pdf_file_path: str | Path,
    *,
    source_id: str | None = None,
    document_type: str | None = None,
) -> list[ExtractedPDFPage]:
    """Extract text from a PDF and return one record per page."""

    pdf_path = Path(pdf_file_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF path is not a file: {pdf_path}")

    reader_cls = _import_pdf_reader()

    try:
        reader = reader_cls(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - keep PDF parse failures readable
        raise PDFProcessingError(f"Failed to open PDF {pdf_path.name}: {exc}") from exc

    inferred_source_id = source_id or pdf_path.stem
    inferred_document_type = document_type or pdf_path.stem
    total_pages = len(reader.pages)
    extracted_pages: list[ExtractedPDFPage] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - page-specific extraction should not stop the run
            raise PDFProcessingError(
                f"Failed to extract page {index} from {pdf_path.name}: {exc}"
            ) from exc

        metadata: dict[str, Any] = {
            "filename": pdf_path.name,
            "source_id": inferred_source_id,
            "document_type": inferred_document_type,
            "page_number": index,
            "total_pages": total_pages,
        }
        extracted_pages.append(
            ExtractedPDFPage(
                text=text,
                filename=pdf_path.name,
                source_id=inferred_source_id,
                document_type=inferred_document_type,
                page_number=index,
                metadata=metadata,
            )
        )

    return extracted_pages
