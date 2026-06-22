"""RAG document ingestion command."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from backend.app.config import settings  # noqa: E402
from backend.app.rag.chunking import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_document_pages,
)
from backend.app.rag.embeddings import EmbeddingProvider, HashEmbeddingProvider  # noqa: E402
from backend.app.rag.vector_store import save_faiss_vector_store  # noqa: E402
from backend.app.utils.pdf_loader import load_pdf_directory  # noqa: E402


class RagIngestionError(ValueError):
    """Raised when ingestion cannot proceed."""


def ingest_documents(
    *,
    raw_documents_dir: str | Path = settings.raw_documents_dir,
    vector_store_dir: str | Path = settings.vector_store_dir,
    embedding_provider: EmbeddingProvider | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Load PDFs, chunk text, embed chunks, and persist a FAISS vector store."""

    documents_path = Path(raw_documents_dir)
    pdf_paths = sorted(documents_path.glob("*.pdf"))
    if not pdf_paths:
        raise RagIngestionError(
            f"No PDF documents found in {documents_path}. Add public insurance PDFs first."
        )

    pages = load_pdf_directory(documents_path)
    chunks = chunk_document_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise RagIngestionError(f"No text chunks produced from PDFs in {documents_path}.")

    provider = embedding_provider or HashEmbeddingProvider()
    embeddings = provider.embed_texts([chunk.content for chunk in chunks])
    summary = save_faiss_vector_store(chunks, embeddings, vector_store_dir=vector_store_dir)

    return {
        "document_count": len(pdf_paths),
        "page_count": len(pages),
        **summary,
    }


def main() -> None:
    try:
        summary = ingest_documents()
    except RagIngestionError as exc:
        print(f"RAG ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("RAG ingestion complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
