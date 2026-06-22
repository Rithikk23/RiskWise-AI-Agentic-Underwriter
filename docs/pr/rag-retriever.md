# Stage 6 PR Draft: Add Source-Cited RAG Retriever

## Title

Stage 6: Add source-cited RAG retriever

## Summary

This PR adds retrieval over the persisted FAISS vector store. It loads the FAISS index and chunk metadata, embeds a query with a mockable provider, supports guideline query types, and returns source-cited context records with content, source, page, score, chunk ID, and document type.

## Changed Files

- `backend/app/rag/retriever.py`
- `backend/app/rag/vector_store.py`
- `tests/test_rag.py`
- `docs/pr/rag-retriever.md`

## Testing

- `.venv/bin/python -m compileall backend`
- `.venv/bin/python -m pytest tests/test_project_structure.py tests/test_data_loader.py tests/test_fraud_model.py tests/test_rag.py`
- `.venv/bin/python -m ruff check .`
- Local smoke test: `retrieve_context("What should happen during claim handling?", top_k=3, query_type="claim_handling")` returned three source-cited chunks from the generated FAISS store.

## Risks

- Retrieval quality is limited by the deterministic hash embedding provider until production embeddings are configured.
- Generated files under `data/vector_store/` remain local-only and ignored by Git.
- Missing vector store files now fail with an actionable error, but users still need to run ingestion after adding public PDFs.

## Acceptance Criteria

- `retrieve_context(query, top_k=5)` returns stable structured results.
- Results include content, source, page, score, chunk ID, and document type.
- Guideline query types are supported and validated.
- Missing vector store files produce an actionable error.
- Tests use fake vector store data and do not require live LLM or embedding API calls.

## Rollback Notes

Revert `backend/app/rag/retriever.py`, the vector store loader addition, related tests, and this PR draft. Existing ingestion code and generated local vector stores can remain in place.
