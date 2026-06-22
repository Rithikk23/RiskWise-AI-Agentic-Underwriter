"""Text cleaning and chunking helpers for public insurance documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentPage:
    """Extracted text from one PDF page."""

    source: str
    page: int
    text: str
    document_type: str = "insurance_guideline"


@dataclass(frozen=True)
class DocumentChunk:
    """Chunked text and source metadata for vector storage."""

    content: str
    metadata: dict[str, Any]


def clean_document_text(text: str) -> str:
    """Normalize PDF text while preserving readable sentence order."""

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character chunks."""

    cleaned_text = clean_document_text(text)
    if not cleaned_text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunks.append(cleaned_text[start:end].strip())
        if end == len(cleaned_text):
            break
        start = end - chunk_overlap

    return chunks


def chunk_document_pages(
    pages: list[DocumentPage],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Chunk extracted pages and attach retrieval metadata."""

    document_chunks: list[DocumentChunk] = []
    for page in pages:
        page_chunks = chunk_text(
            page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for index, content in enumerate(page_chunks):
            chunk_id = f"{page.source}_page{page.page}_chunk{index}"
            document_chunks.append(
                DocumentChunk(
                    content=content,
                    metadata={
                        "source": page.source,
                        "page": page.page,
                        "chunk_id": chunk_id,
                        "document_type": page.document_type,
                    },
                )
            )

    return document_chunks
