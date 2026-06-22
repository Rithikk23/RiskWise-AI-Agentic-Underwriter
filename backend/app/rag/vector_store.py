"""FAISS vector store persistence for RAG chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.app.config import settings
from backend.app.rag.chunking import DocumentChunk

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class VectorStoreError(ValueError):
    """Raised when vector store inputs are invalid."""


def save_faiss_vector_store(
    chunks: list[DocumentChunk],
    embeddings: np.ndarray,
    *,
    vector_store_dir: str | Path = settings.vector_store_dir,
) -> dict[str, Any]:
    """Save a FAISS index and chunk metadata to disk."""

    if not chunks:
        raise VectorStoreError("Cannot save a vector store with no chunks.")
    if embeddings.ndim != 2:
        raise VectorStoreError("Embeddings must be a two-dimensional array.")
    if embeddings.shape[0] != len(chunks):
        raise VectorStoreError("Embedding row count must match chunk count.")

    path = Path(vector_store_dir)
    path.mkdir(parents=True, exist_ok=True)

    vectors = np.asarray(embeddings, dtype=np.float32)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, str(path / INDEX_FILENAME))

    metadata = {
        "index_type": "faiss.IndexFlatL2",
        "embedding_dimension": dimension,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    }
    (path / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "index_path": str(path / INDEX_FILENAME),
        "metadata_path": str(path / METADATA_FILENAME),
        "chunk_count": len(chunks),
        "embedding_dimension": dimension,
    }


def load_vector_store_metadata(
    vector_store_dir: str | Path = settings.vector_store_dir,
) -> dict[str, Any]:
    """Load persisted chunk metadata."""

    path = Path(vector_store_dir) / METADATA_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))
