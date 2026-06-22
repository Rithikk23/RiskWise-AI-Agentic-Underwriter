"""Source-cited retrieval over the local FAISS vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.config import settings
from backend.app.rag.embeddings import EmbeddingProvider, HashEmbeddingProvider
from backend.app.rag.vector_store import (
    INDEX_FILENAME,
    METADATA_FILENAME,
    load_faiss_index,
    load_vector_store_metadata,
)

GUIDELINE_QUERY_EXPANSIONS = {
    "claim_handling": "claim handling settlement documentation timely fair review denial payment",
    "coverage": "coverage liability collision comprehensive deductible limits exclusions policy",
    "fraud": "fraud misrepresentation staged accident investigation suspicious claim",
    "compliance": "regulation compliance unfair claims settlement practices consumer protection",
    "underwriting": "underwriting risk premium driver vehicle policy eligibility",
}


class RagRetrievalError(ValueError):
    """Raised when retrieval cannot proceed."""


@dataclass(frozen=True)
class RetrievalResult:
    """A chunk returned by the retriever with citation metadata."""

    content: str
    source: str
    page: int | None
    score: float
    chunk_id: str | None = None
    document_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "document_type": self.document_type,
        }


def retrieve_context(
    query: str,
    *,
    top_k: int = 5,
    query_type: str | None = None,
    vector_store_dir: str | Path = settings.vector_store_dir,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[dict[str, Any]]:
    """Retrieve source-cited context chunks for an underwriting question."""

    results = retrieve_context_results(
        query,
        top_k=top_k,
        query_type=query_type,
        vector_store_dir=vector_store_dir,
        embedding_provider=embedding_provider,
    )
    return [result.to_dict() for result in results]


def retrieve_context_results(
    query: str,
    *,
    top_k: int = 5,
    query_type: str | None = None,
    vector_store_dir: str | Path = settings.vector_store_dir,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievalResult]:
    """Retrieve typed result objects for callers that prefer dataclasses."""

    normalized_query = query.strip()
    if not normalized_query:
        raise RagRetrievalError("Query must not be empty.")
    if top_k <= 0:
        raise RagRetrievalError("top_k must be positive.")

    store_dir = Path(vector_store_dir)
    _ensure_vector_store_exists(store_dir)

    metadata = load_vector_store_metadata(store_dir)
    chunks = _metadata_chunks(metadata)
    if not chunks:
        raise RagRetrievalError(f"No chunks found in vector store metadata at {store_dir}.")

    index = load_faiss_index(store_dir)
    if index.ntotal != len(chunks):
        raise RagRetrievalError(
            f"Vector store mismatch at {store_dir}: index has {index.ntotal} vectors "
            f"but metadata has {len(chunks)} chunks."
        )

    provider = embedding_provider or HashEmbeddingProvider(dimension=index.d)
    retrieval_query = _build_retrieval_query(normalized_query, query_type=query_type)
    query_vector = provider.embed_texts([retrieval_query])
    if query_vector.shape != (1, index.d):
        raise RagRetrievalError(
            f"Embedding dimension mismatch: expected {index.d}, got {query_vector.shape[1]}."
        )

    limit = min(top_k, len(chunks))
    distances, indices = index.search(np.asarray(query_vector, dtype=np.float32), limit)
    results: list[RetrievalResult] = []
    for distance, index_position in zip(distances[0], indices[0], strict=True):
        if index_position < 0:
            continue
        chunk = chunks[int(index_position)]
        chunk_metadata = chunk.get("metadata", {})
        results.append(
            RetrievalResult(
                content=str(chunk.get("content", "")),
                source=str(chunk_metadata.get("source", "unknown")),
                page=_optional_int(chunk_metadata.get("page")),
                score=_distance_to_score(float(distance)),
                chunk_id=_optional_str(chunk_metadata.get("chunk_id")),
                document_type=_optional_str(chunk_metadata.get("document_type")),
            )
        )

    return results


def _ensure_vector_store_exists(store_dir: Path) -> None:
    missing = [
        filename
        for filename in (INDEX_FILENAME, METADATA_FILENAME)
        if not (store_dir / filename).exists()
    ]
    if missing:
        missing_files = ", ".join(missing)
        raise RagRetrievalError(
            f"Missing vector store files in {store_dir}: {missing_files}. "
            "Run `python backend/app/rag/ingest.py` after adding public PDFs."
        )


def _metadata_chunks(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = metadata.get("chunks")
    if not isinstance(chunks, list):
        raise RagRetrievalError("Vector store metadata is missing a `chunks` list.")
    return chunks


def _build_retrieval_query(query: str, *, query_type: str | None) -> str:
    if query_type is None:
        return query
    try:
        expansion = GUIDELINE_QUERY_EXPANSIONS[query_type]
    except KeyError as exc:
        supported = ", ".join(sorted(GUIDELINE_QUERY_EXPANSIONS))
        raise RagRetrievalError(
            f"Unsupported query_type `{query_type}`. Supported values: {supported}."
        ) from exc
    return f"{query} {expansion}"


def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
