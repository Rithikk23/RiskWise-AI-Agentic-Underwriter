# Stage 5 PR Draft: Add Public Document RAG Ingestion Pipeline

## Title

Stage 5: Add public document RAG ingestion pipeline

## Summary

This PR adds the RAG ingestion pipeline for public insurance PDFs. It extracts PDF text, cleans and chunks document pages, creates deterministic local embeddings by default, and writes a FAISS index plus metadata under `data/vector_store/`.

## Changed Files

- `backend/app/rag/ingest.py`
- `backend/app/rag/chunking.py`
- `backend/app/rag/embeddings.py`
- `backend/app/rag/vector_store.py`
- `backend/app/utils/pdf_loader.py`
- `tests/test_rag.py`
- `docs/pr/rag-ingestion.md`

## Testing

- `.venv/bin/python -m compileall backend`
- `.venv/bin/python -m pytest tests/test_project_structure.py tests/test_data_loader.py tests/test_fraud_model.py tests/test_rag.py`
- `.venv/bin/python -m ruff check .`
- `.venv/bin/python backend/app/rag/ingest.py` was smoke-tested against a local public corpus and produced 311 chunks across 13 ignored raw documents.
- Empty-folder behavior is covered by `tests/test_rag.py`.

## Risks

- `data/raw_documents/` contains local-only public source PDFs and converted public HTML pages; these files are ignored by Git and must be added locally before real ingestion.
- `data/vector_store/index.faiss` and `data/vector_store/metadata.json` are generated locally and ignored by Git.
- The default hash embedding provider is deterministic and offline-friendly; production-quality semantic retrieval can switch to OpenAI embeddings in a later configuration pass.

## Acceptance Criteria

- Empty document folders fail with a clear error.
- PDF text extraction works against a generated fixture PDF.
- Chunk metadata includes `source`, `page`, `chunk_id`, and `document_type`.
- FAISS index and metadata files are saved for non-empty document sets.
- Tests do not require live LLM or embedding API calls.

## Rollback Notes

Revert the RAG ingestion modules, tests, and this PR draft. Delete any local generated vector store files if desired.
