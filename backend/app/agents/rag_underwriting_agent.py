"""RAG underwriting agent for public guidance retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.app.rag.retriever import RagRetrievalError, retrieve_context
from backend.app.schemas import Citation, RagUnderwritingOutput

Retriever = Callable[..., list[dict[str, Any]]]


def run_rag_underwriting_agent(
    question: str,
    *,
    retriever: Retriever = retrieve_context,
    top_k: int = 5,
    query_type: str | None = None,
) -> RagUnderwritingOutput:
    """Retrieve public underwriting or claim guidance for a question."""

    selected_query_type = query_type or infer_query_type(question)
    try:
        contexts = retriever(question, top_k=top_k, query_type=selected_query_type)
    except RagRetrievalError as exc:
        return RagUnderwritingOutput(
            query=question,
            query_type=selected_query_type,
            guidance_summary=f"No public guidance could be retrieved: {exc}",
            citations=[],
        )

    citations = [_citation_from_context(context) for context in contexts]
    if citations:
        guidance_summary = (
            f"Retrieved {len(citations)} public guidance excerpts for "
            f"{selected_query_type.replace('_', ' ')} review."
        )
    else:
        guidance_summary = "No public guidance excerpts were retrieved."

    return RagUnderwritingOutput(
        query=question,
        query_type=selected_query_type,
        guidance_summary=guidance_summary,
        citations=citations,
    )


def infer_query_type(question: str) -> str:
    """Infer the retriever query type from a plain-language question."""

    normalized = question.lower()
    if any(term in normalized for term in ("claim", "settlement", "payment", "denial")):
        return "claim_handling"
    if any(term in normalized for term in ("fraud", "suspicious", "misrepresentation")):
        return "fraud"
    if any(term in normalized for term in ("coverage", "deductible", "liability", "limit")):
        return "coverage"
    if any(term in normalized for term in ("compliance", "regulation", "fair")):
        return "compliance"
    return "underwriting"


def _citation_from_context(context: dict[str, Any]) -> Citation:
    content = str(context.get("content", ""))
    return Citation(
        source=str(context.get("source", "unknown")),
        page=_optional_int(context.get("page")),
        score=_optional_float(context.get("score")),
        excerpt=content[:500],
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
