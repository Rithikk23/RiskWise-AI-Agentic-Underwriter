"""Compliance support agent for source-backed recommendations."""

from __future__ import annotations

from backend.app.schemas import ComplianceOutput, RagUnderwritingOutput, RiskAssessmentOutput


def run_compliance_agent(
    rag_guidance: RagUnderwritingOutput,
    risk_assessment: RiskAssessmentOutput,
) -> ComplianceOutput:
    """Check that downstream decisions have source-backed public guidance."""

    issues: list[str] = []
    citations_checked = len(rag_guidance.citations)
    if citations_checked == 0:
        issues.append("No public source citations retrieved")

    if risk_assessment.risk_level == "High" and citations_checked < 2:
        issues.append("High-risk decisions require at least two supporting citations")

    is_supported = not issues
    return ComplianceOutput(
        status="Compliant" if is_supported else "Needs Review",
        is_supported=is_supported,
        issues=issues,
        citations_checked=citations_checked,
    )
