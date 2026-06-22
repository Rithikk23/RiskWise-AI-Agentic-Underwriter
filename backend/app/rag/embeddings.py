"""Embedding providers for RAG ingestion."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

DEFAULT_EMBEDDING_DIMENSION = 128
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class EmbeddingProvider(Protocol):
    """Protocol for embedding text batches."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return one embedding vector per text."""


class HashEmbeddingProvider:
    """Deterministic local embedding provider for offline ingestion and tests."""

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = TOKEN_PATTERN.findall(text.lower())
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row_index, bucket] += sign

            norm = np.linalg.norm(vectors[row_index])
            if norm > 0:
                vectors[row_index] = vectors[row_index] / norm

        return vectors


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider, loaded lazily so local tests can run offline."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("Install openai to use OpenAIEmbeddingProvider.") from exc

        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)
