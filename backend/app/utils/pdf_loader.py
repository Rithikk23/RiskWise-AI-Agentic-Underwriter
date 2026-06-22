"""PDF loading utilities for public insurance documents."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from backend.app.rag.chunking import DocumentPage, clean_document_text


class PdfTextExtractionError(ValueError):
    """Raised when a PDF cannot be read or contains no extractable text."""


def load_pdf_pages(
    pdf_path: str | Path,
    *,
    document_type: str = "insurance_guideline",
) -> list[DocumentPage]:
    """Extract text from every readable page of a PDF."""

    path = Path(pdf_path)
    reader = PdfReader(path)
    pages: list[DocumentPage] = []

    for index, page in enumerate(reader.pages, start=1):
        text = clean_document_text(page.extract_text() or "")
        if not text:
            continue
        pages.append(
            DocumentPage(
                source=path.name,
                page=index,
                text=text,
                document_type=document_type,
            )
        )

    if not pages:
        raise PdfTextExtractionError(f"No extractable text found in {path}.")

    return pages


def load_pdf_directory(
    directory: str | Path,
    *,
    document_type: str = "insurance_guideline",
) -> list[DocumentPage]:
    """Load all PDF pages from a directory."""

    path = Path(directory)
    pdf_paths = sorted(path.glob("*.pdf"))
    pages: list[DocumentPage] = []

    for pdf_path in pdf_paths:
        pages.extend(load_pdf_pages(pdf_path, document_type=document_type))

    return pages
