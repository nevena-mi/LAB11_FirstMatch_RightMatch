Revised Implementation Plan: Podcast + PDF RAG Lab
## Checklist
- [x] Step 1 - Project Setup
- [x] Step 2 - Podcast Transcription
- [x] Step 3 - PDF Processing
- [x] Step 4 - Common Document Representation
- [x] Step 5 - Chunking
- [x] Step 6 - Embeddings
- [x] Step 7 - Vector Store
- [x] Step 8 - Baseline Retrieval
- [x] Step 9 - LLM Relevance Scoring
- [x] Step 10 - Dedicated Reranker
- [x] Step 11 - Metadata Filtering
- [ ] Step 12 - Complete RAG Pipeline
- [ ] Step 13 - Evaluation and lab_proof.md
Summary
Build the lab in small, independently testable steps. The notebook stays as the orchestration and inspection surface, while src/ holds reusable pipeline logic. The main change from the previous plan is an early transcription stage: the podcast must first be converted to text with Whisper, then treated as a source document alongside the PDFs.
Step-by-Step Order
1. Project Setup
Module(s): src/utils.py, src/rag_pipeline.py, notebook bootstrap cells
Input: Existing repo, .env, source files in sources/
Output: Confirmed project skeleton, loaded env vars, and a notebook that can import from src/
What to do: Define the module boundaries, verify OPENAI_API_KEY and other required keys, and confirm the notebook can resolve local imports and source paths.
Test before continuing: Run a minimal notebook import check and a small env validation check. Confirm the source PDFs and podcast file are discoverable from the repo.
2. Podcast Transcription
Module(s): src/transcription.py new
Input: Raw podcast audio file
Output: Transcript text file or markdown file, plus metadata record
Metadata to capture: source, speaker if detectable, duration, filename, optionally language, transcription_model
What to do: Transcribe the audio using Whisper, save the transcript to disk, and attach metadata in a structured form.
Test before continuing: Run transcription on the podcast alone, inspect the saved transcript, and verify metadata is present and readable without any other pipeline steps.
3. PDF Processing
Module(s): src/pdf_processor.py
Input: EU AI Act PDF, Trustworthy AI PDF
Output: Extracted text records, ideally one record per page or logical page group, with metadata
Metadata to preserve: filename, page_number, document_type, and any other useful source markers
What to do: Extract text from each PDF in a way that preserves page identity and source identity.
Test before continuing: Extract a few sample pages from each PDF and confirm the text and metadata match the correct source/page.
4. Common Document Representation
Module(s): src/utils.py or a small dedicated helper inside src/ if needed
Input: Transcript file + PDF extraction output
Output: A single normalized document structure for all sources
What to do: Convert the podcast transcript and PDF pages into the same schema so later stages do not care about the original source type.
Recommended schema: text, metadata, source_id, document_type, chunkable_text or equivalent
Test before continuing: Load one podcast record and one PDF record, normalize both, and confirm they expose the same fields and compatible metadata keys.

## Completed

### Document normalization layer
- Added NormalizedDocument dataclass as common representation.
- Implemented normalization for:
  - extracted PDF pages
  - podcast transcription results
- Verified that different document sources produce the same schema.

## Next
- Implement document chunking layer.
- Split normalized documents into smaller retrieval units.

5. Chunking
Module(s): src/chunking.py
Input: Normalized documents
Output: Chunked documents with inherited metadata and chunk identifiers
What to do: Split each document into chunks while preserving source metadata, page number where relevant, and transcript origin.
Test before continuing: Chunk one PDF page and one transcript section, then inspect chunk boundaries and metadata propagation.
6. Embeddings
Module(s): src/embeddings.py
Input: Chunked documents and query text
Output: Embedding vectors for chunks and queries
What to do: Generate embeddings for all chunks and define the same embedding path for future queries.
Test before continuing: Embed a small sample of chunks and a sample query, verify vector shapes/types, and confirm repeated calls are stable.
7. Vector Store
Module(s): src/vector_store.py
Input: Embedded chunks with metadata
Output: Stored vectors and retrieval-ready records in Pinecone
What to do: Upsert chunk embeddings into Pinecone with their metadata and retrieval IDs.
Test before continuing: Insert a tiny batch, query the index, and verify returned IDs and metadata match the inserted chunks.
8. Baseline Retrieval
Module(s): src/retrieval.py
Input: User query and vector store
Output: Top-k candidate chunks with scores and metadata
What to do: Implement plain similarity retrieval as the baseline control path.
Test before continuing: Run one EU AI Act query and one podcast-related query, then verify the results are sensible and traceable back to source documents.
9. LLM Relevance Scoring
Module(s): src/relevance_scoring.py
Input: Query plus baseline retrieved candidates
Output: Candidate chunks with LLM relevance scores and traceability preserved
What to do: Add a scoring pass that asks the LLM to judge whether a retrieved chunk actually answers the query.
Test before continuing: Compare the scored output to the baseline output on the trustworthy AI query and verify the retrieval scores, relevance scores, and traceability fields are all present.
10. Dedicated Reranker
Module(s): src/reranker.py
Input: Query plus candidate chunks
Output: Reranked candidates from Cohere
What to do: Add the dedicated reranker as the production-style relevance improvement stage.
Test before continuing: Feed the same baseline candidates through the reranker and verify the ranking changes are reflected in the output order.
11. Metadata Filtering
Module(s): src/retrieval.py or src/vectorstore.py
Input: Query plus optional filter criteria
Output: Filtered candidate set before similarity or reranking
What to do: Allow filtering by source, document type, or other metadata before or during retrieval.
Test before continuing: Filter to only EU AI Act content, then run a retrieval query and confirm podcast chunks are excluded.
12. Complete RAG Pipeline
Module(s): src/rag_pipeline.py
Input: Query, retriever, reranker, metadata filters, prompt template
Output: Final answer plus supporting evidence chunks
What to do: Wire together retrieval, optional scoring, reranking, and answer generation into one callable pipeline.
Test before continuing: Run a full end-to-end query and verify the returned answer includes cited evidence that can be traced to the retrieved chunks.
13. Evaluation and lab_proof.md
Module(s): notebook, lab_proof.md
Input: Baseline and reranked outputs
Output: Completed proof document with query, retrieved evidence, final answer, and limitation
What to do: Compare baseline vs improved retrieval, document the evidence used, and write up one failure case or limitation.
Test before continuing: Fill out lab_proof.md for at least one EU AI Act question and one podcast-derived question, then verify every claim is backed by retrieved source text.
Recommended Notebook vs src/ Split
Notebook should contain:setup and environment checks
exploratory runs
visual inspection of chunks and retrieval results
manual comparison of baseline vs reranked outputs
final lab evidence and narrative

src/ should contain:transcription, PDF parsing, normalization, chunking, embeddings, vector store access, retrieval, reranking, and the pipeline wrapper

Assumptions
The podcast audio file is already available locally in the repo or a known source path.
The lab is intentionally small, so the implementation should prefer straightforward functions over heavy abstractions.
A single shared document schema is enough for PDFs and transcript text.
Each step should be kept narrow enough to validate with a quick notebook smoke test before moving on.
