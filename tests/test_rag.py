from pathlib import Path

import numpy as np
import pytest

from backend.app.rag.chunking import DocumentChunk, chunk_text
from backend.app.rag.embeddings import HashEmbeddingProvider
from backend.app.rag.ingest import RagIngestionError, ingest_documents
from backend.app.rag.retriever import RagRetrievalError, retrieve_context
from backend.app.rag.vector_store import (
    INDEX_FILENAME,
    METADATA_FILENAME,
    load_vector_store_metadata,
    save_faiss_vector_store,
)
from backend.app.utils.pdf_loader import load_pdf_pages


def test_chunk_text_uses_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=5, chunk_overlap=2)

    assert chunks == ["abcde", "defgh", "ghij"]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("abc", chunk_size=3, chunk_overlap=3)


def test_hash_embedding_provider_is_deterministic() -> None:
    provider = HashEmbeddingProvider(dimension=16)

    first = provider.embed_texts(["claim handling guideline"])
    second = provider.embed_texts(["claim handling guideline"])

    assert first.shape == (1, 16)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_save_faiss_vector_store_writes_index_and_metadata(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            content="claim handling rules",
            metadata={
                "source": "sample.pdf",
                "page": 1,
                "chunk_id": "sample_page1_chunk0",
                "document_type": "insurance_guideline",
            },
        )
    ]
    embeddings = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)

    summary = save_faiss_vector_store(chunks, embeddings, vector_store_dir=tmp_path)
    metadata = load_vector_store_metadata(tmp_path)

    assert (tmp_path / INDEX_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()
    assert summary["chunk_count"] == 1
    assert metadata["chunks"][0]["metadata"]["source"] == "sample.pdf"


def test_load_pdf_pages_extracts_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample_guideline.pdf"
    _write_text_pdf(pdf_path, "Claim handling guidelines require timely review.")

    pages = load_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].source == "sample_guideline.pdf"
    assert pages[0].page == 1
    assert "Claim handling guidelines" in pages[0].text


def test_ingest_documents_raises_for_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(RagIngestionError, match="No PDF documents found"):
        ingest_documents(raw_documents_dir=tmp_path, vector_store_dir=tmp_path / "vector_store")


def test_ingest_documents_creates_vector_store(tmp_path: Path) -> None:
    raw_documents_dir = tmp_path / "raw_documents"
    vector_store_dir = tmp_path / "vector_store"
    raw_documents_dir.mkdir()
    _write_text_pdf(
        raw_documents_dir / "auto_claims.pdf",
        "Auto insurance claim handling guidelines require documentation and fair review.",
    )

    summary = ingest_documents(
        raw_documents_dir=raw_documents_dir,
        vector_store_dir=vector_store_dir,
        chunk_size=40,
        chunk_overlap=10,
    )
    metadata = load_vector_store_metadata(vector_store_dir)

    assert summary["document_count"] == 1
    assert summary["page_count"] == 1
    assert summary["chunk_count"] >= 1
    assert metadata["chunks"][0]["metadata"]["document_type"] == "insurance_guideline"


def test_retrieve_context_returns_source_cited_results(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            content="Claim handling guidelines require fair review.",
            metadata={
                "source": "claims.pdf",
                "page": 2,
                "chunk_id": "claims_page2_chunk0",
                "document_type": "insurance_guideline",
            },
        ),
        DocumentChunk(
            content="Underwriting guidelines discuss premium factors.",
            metadata={
                "source": "underwriting.pdf",
                "page": 4,
                "chunk_id": "underwriting_page4_chunk0",
                "document_type": "insurance_guideline",
            },
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    save_faiss_vector_store(chunks, embeddings, vector_store_dir=tmp_path)

    results = retrieve_context(
        "How should this claim be reviewed?",
        top_k=1,
        query_type="claim_handling",
        vector_store_dir=tmp_path,
        embedding_provider=_StaticEmbeddingProvider([[1.0, 0.0]]),
    )

    assert results == [
        {
            "content": "Claim handling guidelines require fair review.",
            "source": "claims.pdf",
            "page": 2,
            "score": 1.0,
            "chunk_id": "claims_page2_chunk0",
            "document_type": "insurance_guideline",
        }
    ]


def test_retrieve_context_reports_missing_vector_store(tmp_path: Path) -> None:
    with pytest.raises(RagRetrievalError, match="Run `python backend/app/rag/ingest.py`"):
        retrieve_context("claim handling", vector_store_dir=tmp_path)


def test_retrieve_context_rejects_unknown_query_type(tmp_path: Path) -> None:
    chunks = [
        DocumentChunk(
            content="Coverage guidelines",
            metadata={
                "source": "coverage.pdf",
                "page": 1,
                "chunk_id": "coverage_page1_chunk0",
                "document_type": "insurance_guideline",
            },
        )
    ]
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    save_faiss_vector_store(chunks, embeddings, vector_store_dir=tmp_path)

    with pytest.raises(RagRetrievalError, match="Unsupported query_type"):
        retrieve_context(
            "coverage",
            query_type="medical_guideline",
            vector_store_dir=tmp_path,
            embedding_provider=_StaticEmbeddingProvider([[1.0, 0.0]]),
        )


class _StaticEmbeddingProvider:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.resize(self.vectors, (len(texts), self.vectors.shape[1])).astype(np.float32)


def _write_text_pdf(path: Path, text: str) -> None:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n{body}\nendobj\n".encode())

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode()
    )
    path.write_bytes(bytes(content))
