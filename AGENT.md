# Project Overview

This project is a small educational RAG lab focused on relevance scoring and rerankers using the EU AI Act PDF, a Trustworthy AI PDF, and a podcast source.

Current pipeline architecture:

PDF / Audio
↓
Extraction
↓
Normalization
↓
Chunking
↓
Embeddings
↓
Vector Store
↓
Retrieval
↓
LLM Relevance Scoring
↓
LLM Response (planned)

Current implementation status:
- Step 1 is complete: project setup, environment validation, and notebook import scaffolding.
- Step 2 is complete: podcast preprocessing and Whisper transcription with saved transcript and metadata.
- Step 3 is complete: PDF extraction into page-level records.
- Step 4 is complete: normalization into a shared document schema.
- Step 5 is complete: deterministic character-based chunking with overlap.
- Step 6 is complete: OpenAI-based chunk embeddings with preserved traceability.
- Step 7 is complete and validated: Pinecone vector store integration, metadata sanitization, and notebook verification.
- Step 8 is complete and validated: baseline Pinecone retrieval with stable traceability and repeated-query consistency.
- Step 9 is complete and validated: LLM relevance scoring over baseline retrieval candidates with preserved traceability.
- Later stages are not implemented yet.

# Architecture

Current pipeline:

PDF / Audio
↓
Extraction
↓
Normalization
↓
Chunking
↓
Embeddings
↓
Vector Store
↓
Retrieval
↓
LLM Relevance Scoring
↓
LLM Response (planned)

The implemented pipeline currently includes embeddings, a Pinecone vector store layer, baseline retrieval, and an LLM relevance scoring pass. Later stages are intentionally deferred to keep the lab incremental and easy to reason about.

# Implemented Components

## `src/pdf_processor.py`
Extracts one record per PDF page.
- Purpose: turn each source PDF into traceable page-level records.
- Output: `ExtractedPDFPage` objects.
- Traceability: preserves filename, source ID, document type, page number, and total pages.

## `src/transcription.py`
Transcribes the podcast audio and persists transcript artifacts.
- Purpose: convert the podcast into text for use as a RAG source.
- Output: `TranscriptionResult` objects.
- Traceability: stores the transcript text and metadata including source, filename, model, language when available, duration when available, and speaker when available.

## `src/normalization.py`
Converts PDF pages and transcription results into a shared document schema.
- Purpose: provide one common representation for later chunking and retrieval.
- Output: `NormalizedDocument` objects.
- Traceability: preserves upstream metadata as-is and adds a stable document identifier.

## `src/chunking.py`
Splits normalized documents into retrieval-ready chunks.
- Purpose: create deterministic chunks from normalized documents.
- Output: `Chunk` objects.
- Traceability: preserves document IDs, source IDs, document types, and copied metadata, while adding chunk-specific fields.

## `src/embeddings.py`
Converts chunks into vector representations for later retrieval.
- Purpose: turn retrieval-ready chunks into embedded records without changing their traceability fields.
- Output: `EmbeddedChunk` objects.
- Traceability: preserves chunk IDs, document IDs, source IDs, document types, and chunk metadata.

## `src/vector_store.py`
Wraps Pinecone index creation, upserts, and similarity queries.
- Purpose: store embedded chunks and verify vectors can be retrieved by similarity.
- Output: Pinecone upsert/query responses.
- Traceability: stores `document_id`, `source_id`, `document_type`, `text`, and copied chunk metadata alongside each vector.

## `src/retrieval.py`
Wraps query embedding and Pinecone similarity search for baseline retrieval.
- Purpose: embed a user query and return the most relevant stored chunks.
- Output: `RetrievedChunk` objects.
- Traceability: preserves `chunk_id`, `document_id`, `source_id`, `document_type`, `text`, `metadata`, and similarity score.

## `src/relevance_scoring.py`
Scores baseline retrieval results with an LLM while keeping traceability intact.
- Purpose: ask the model how relevant each retrieved chunk is to the query.
- Output: `ScoredChunk` objects.
- Traceability: preserves the original chunk and document identifiers, source IDs, document types, metadata, and the original Pinecone similarity score.

## Validation result
- A notebook validation cell was appended to score the trustworthiness query against the baseline retrieval results.
- Local stubbed-client validation confirmed that `score_relevance()` preserves chunk IDs, document IDs, source IDs, document types, metadata, and retrieval scores.
- The validation path also confirmed that relevance scores and optional reasoning are returned together.
- Repeated calls with the same input produced the same chunk order and the same parsed scores in the local validation check.

## Implementation decisions and rationale
- The OpenAI Responses API was used with structured JSON output so the scoring layer stays easy to parse and debug.
- The scoring model is configurable, with `gpt-4o-mini` as the default for the lab.
- The model is instructed to return scores from 0.0 to 1.0 to keep the output simple and comparable.
- The output preserves the baseline retrieval order rather than reordering chunks, because reranking is a separate later step.
- `seed=42` and `temperature=0` were chosen to keep repeated runs as stable as possible for an educational lab.

# Embeddings

The embedding layer is the bridge between chunked documents and the later retrieval stack.
- It keeps the implementation isolated from vector databases and search logic.
- It uses the OpenAI Embeddings API with lazy client initialization so the module stays reusable and does not require global client setup at import time.
- The default model is `text-embedding-3-small`.
- The layer does not truncate or alter chunk text before embedding.
- Metadata is copied into `EmbeddedChunk` records so original traceability is preserved without mutating upstream chunk objects.

## `EmbeddedChunk`
- Purpose: represent a chunk after embedding has been added.
- Important fields:
  - `chunk_id`
  - `document_id`
  - `text`
  - `embedding`
  - `metadata`
  - `source_id`
  - `document_type`
- Used by: `src/embeddings.py` and later vector store / retrieval stages.

## Embedding functions
- `embed_chunk(chunk)`: embeds one `Chunk` and returns one `EmbeddedChunk`.
- `embed_chunks(chunks)`: embeds a list of chunks in order and returns a list of `EmbeddedChunk` objects.
- Both functions preserve chunk identity and metadata.

## Validation result
- Validation was run in the notebook with normalized PDF and transcript documents, then chunking, then embeddings.
- The smoke test produced 24 embedded chunks.
- The embedding dimension was 1536 for every vector.
- The first five embedding values were printed successfully during validation.
- Chunk IDs and document IDs were unchanged after embedding.
- Metadata was preserved exactly from the chunk objects.

## Implementation decisions and rationale
- Lazy OpenAI client creation was used so the module stays import-safe and test-friendly.
- The embedding model is configurable, but the default remains `text-embedding-3-small` for simplicity and consistency.
- The embedding layer remains intentionally small and does not introduce batching abstractions beyond the minimal `embed_chunks` helper.

# Vector Store

The Pinecone layer is kept intentionally small and env-driven.
- It uses the current Pinecone Python SDK.
- It reads all Pinecone settings from environment variables only.
- The API key can come from `PINECONE_API_KEY` or the existing project alias `PINECONE_KEY`.
- The index name, cloud, and region come from `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, and `PINECONE_REGION`.
- The code creates a serverless index when needed and then upserts dense vectors with cosine similarity.
- Each stored record uses the chunk ID as the Pinecone vector ID.
- Metadata is copied from the chunk and augmented with document traceability fields, without mutating the source object.

## `PineconeConfig`
- Purpose: group the required Pinecone settings read from the environment.
- Important fields:
  - `api_key`
  - `index_name`
  - `cloud`
  - `region`
- Used by: `src/vector_store.py` for index creation and access.

## Vector store functions
- `create_index_if_needed(dimension)`: creates the configured index if it does not already exist.
- `get_index(index_name=None)`: returns a Pinecone index handle.
- `upsert_chunk(chunk)`: stores one embedded chunk.
- `upsert_chunks(chunks)`: stores multiple embedded chunks.
- `query_index(query_embedding, top_k=5)`: runs a similarity query and returns top matches with metadata.

## Retrieval functions
- `retrieve(query_embedding, top_k=5)`: queries Pinecone with a dense vector and returns traceable matches.
- `retrieve_query(query_text, top_k=5)`: embeds the query and retrieves the top stored chunks.

## Validation result
- Notebook validation was executed successfully.
- Pinecone index: `ironhack-rag`.
- Vectors uploaded: `24`.
- Query returned the expected chunk IDs.
- `document_id` and `source_id` metadata were preserved in the returned matches.
- Example query: `What are the requirements for trustworthy AI?`
- Top results were retrieved from the trustworthy AI podcast transcript.

## Implementation decisions and rationale
- The wrapper keeps Pinecone-specific logic isolated from embeddings and retrieval.
- Configuration is fully environment-driven to avoid hardcoding deployment details.
- A default cosine metric was used for the dense vector index because the pipeline is using OpenAI dense embeddings.
- A short sleep is used in the notebook validation cell to reduce the chance of querying before Pinecone has indexed the upserted vectors.
- Metadata sanitization removes `None` values before upsert so Pinecone accepts the payload while preserving all usable traceability fields.
- Baseline retrieval stays separate from relevance scoring so the lab can validate retrieval quality before introducing more complex scoring logic.

## Files changed
- `src/relevance_scoring.py`
- `relevance_scoring_rerankers.ipynb`
- `implementation_plan.md`
- `AGENT.md`

# Data Models

## `ExtractedPDFPage`
- Purpose: represent one extracted PDF page.
- Important fields:
  - `text`
  - `filename`
  - `source_id`
  - `document_type`
  - `page_number`
  - `metadata`
- Used by: `src/pdf_processor.py` and `src/normalization.py`.

## `TranscriptionResult`
- Purpose: represent a completed podcast transcription plus saved file paths.
- Important fields:
  - `text`
  - `metadata`
  - `transcript_path`
  - `metadata_path`
- Used by: `src/transcription.py` and `src/normalization.py`.

## `NormalizedDocument`
- Purpose: shared document format for later pipeline stages.
- Important fields:
  - `document_id`
  - `text`
  - `metadata`
  - `source_id`
  - `document_type`
- Used by: `src/normalization.py` and `src/chunking.py`.

## `Chunk`
- Purpose: retrieval-ready slice of a normalized document.
- Important fields:
  - `chunk_id`
  - `text`
  - `metadata`
  - `source_id`
  - `document_type`
  - `document_id`
- Used by: `src/chunking.py` and later embedding / retrieval stages.

## `ScoredChunk`
- Purpose: retrieval result annotated with LLM relevance scoring.
- Important fields:
  - `chunk_id`
  - `document_id`
  - `text`
  - `metadata`
  - `source_id`
  - `document_type`
  - `retrieval_score`
  - `relevance_score`
  - `reasoning`
- Used by: `src/relevance_scoring.py` and the notebook validation cell.

# Design Decisions

- Use dataclasses for the main document types.
- Use `slots=True` where appropriate to keep the objects lightweight and explicit.
- Use deterministic IDs for normalized documents and chunks.
- Use deterministic behavior for embedded chunk ordering and identity preservation.
- Use deterministic Pinecone vector IDs based on chunk IDs.
- Preserve source traceability across every stage.
- Copy metadata into chunk objects instead of mutating upstream metadata.
- Copy metadata into embedded chunk objects instead of mutating upstream chunk metadata.
- Copy metadata into Pinecone vectors instead of mutating source objects.
- Use character-based chunking with overlap to keep the implementation simple and transparent.
- Append notebook cells instead of modifying previous executed cells.
- Build the project in small incremental steps.
- Update `implementation_plan.md` only after successful validation of a step.

# Notebook Conventions

- Existing notebook cells should never be rewritten.
- New functionality should be appended as new cells.
- Previous outputs should be preserved whenever possible.
- The notebook serves as a learning record as well as a smoke-test surface.

# Validation Strategy

Each new feature is validated before the step is marked complete.
- Environment checks verify required API keys are present without exposing values.
- Podcast preprocessing is validated by checking that the output file is smaller and transcribable.
- Transcription is validated by confirming transcript and metadata files are written.
- PDF extraction is validated by checking page counts, metadata, and sample text output.
- Normalization is validated by comparing PDF and transcript objects in the same schema.
- Chunking is validated by checking chunk counts, chunk IDs, text previews, and metadata preservation.
- Embeddings are validated by checking vector dimensions, model choice, chunk identity preservation, and metadata preservation.
- Pinecone integration is validated by confirming index creation, successful upserts, matching retrieved IDs, and preserved metadata.
- LLM relevance scoring is validated by checking the returned relevance scores, preserved retrieval scores, traceability fields, and repeatability on the same query.

# Coding Style

- Keep modules small and focused.
- Use clear docstrings.
- Prefer type hints.
- Favor deterministic behavior.
- Avoid unnecessary dependencies.
- Optimize for educational readability over abstraction or performance.

# Future Roadmap

Remaining stages:
- dedicated reranker
- evaluation

# Instructions for Future Codex Sessions

- Always read `AGENT.md` before making changes.
- Preserve the current architecture and step order.
- Prefer incremental changes over broad refactors.
- Avoid rewriting working code unless a compatibility fix is required.
- Avoid unnecessary refactoring.
- Preserve notebook history by appending new cells instead of replacing old ones.
- Keep `implementation_plan.md` synchronized with completed work.
