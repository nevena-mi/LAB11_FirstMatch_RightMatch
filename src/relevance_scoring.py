"""LLM-based relevance scoring helpers for baseline retrieval results.

This module keeps the scoring pass isolated from retrieval and any future
dedicated reranker. It asks an OpenAI model to judge how relevant each
retrieved chunk is for a query while preserving the original traceability
fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .embeddings import _get_openai_client
from .retrieval import RetrievedChunk

DEFAULT_RELEVANCE_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = (
    "You score retrieved document chunks for relevance to a user query.\n"
    "Return only JSON that matches the provided schema.\n"
    "Score each chunk independently from 0.0 to 1.0.\n"
    "Use a higher score when the chunk directly helps answer the query.\n"
    "Keep reasoning brief and evidence-based."
)

_RELEVANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "chunk_id": {"type": "string"},
                    "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasoning": {"type": "string"},
                },
                "required": ["chunk_id", "relevance_score", "reasoning"],
            },
        }
    },
    "required": ["items"],
}


@dataclass(slots=True)
class ScoredChunk:
    """A retrieved chunk with an LLM relevance score."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    source_id: str
    document_type: str
    retrieval_score: float
    relevance_score: float
    reasoning: str | None = None


def _chunk_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_id": chunk.source_id,
        "document_type": chunk.document_type,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Serialize the query and candidate chunks for the model."""

    payload = {
        "query": query,
        "chunks": [_chunk_payload(chunk) for chunk in chunks],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_scored_items(
    chunks: list[RetrievedChunk],
    raw_response: str,
) -> list[ScoredChunk]:
    """Convert the model response into scored chunk records."""

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse relevance scoring response: {exc}") from exc

    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Relevance scoring response did not include an item list.")

    items_by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("Relevance scoring response contained a non-object item.")
        chunk_id = str(item.get("chunk_id", ""))
        if not chunk_id:
            raise RuntimeError("Relevance scoring response contained an empty chunk_id.")
        items_by_id[chunk_id] = item

    scored_chunks: list[ScoredChunk] = []
    for chunk in chunks:
        item = items_by_id.get(chunk.chunk_id)
        if item is None:
            raise RuntimeError(
                f"Relevance scoring response did not include a score for chunk_id={chunk.chunk_id!r}."
            )

        reasoning = item.get("reasoning")
        scored_chunks.append(
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                source_id=chunk.source_id,
                document_type=chunk.document_type,
                retrieval_score=float(chunk.score),
                relevance_score=float(item["relevance_score"]),
                reasoning=str(reasoning) if reasoning is not None else None,
            )
        )

    return scored_chunks


def score_relevance(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    model: str = DEFAULT_RELEVANCE_MODEL,
    client: Any | None = None,
) -> list[ScoredChunk]:
    """Score retrieved chunks for relevance to a query using an OpenAI model."""

    if not chunks:
        return []

    openai_client = _get_openai_client(client)
    prompt = _build_prompt(query, chunks)

    try:
        response = openai_client.responses.create(
            model=model,
            instructions=_SYSTEM_PROMPT,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "relevance_scoring",
                    "schema": _RELEVANCE_SCHEMA,
                    "strict": True,
                }
            },
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - wrap API/auth/network failures cleanly
        raise RuntimeError(f"Failed to score relevance: {exc}") from exc

    raw_response = getattr(response, "output_text", "") or ""
    if not raw_response:
        raise RuntimeError("Relevance scoring response was empty.")

    return _parse_scored_items(chunks, raw_response)
