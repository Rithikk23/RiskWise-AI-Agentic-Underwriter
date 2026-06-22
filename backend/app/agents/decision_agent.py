"""Final decision agent for underwriting recommendations."""

from __future__ import annotations

from backend.app.schemas import (
    ComplianceOutput,
    DecisionOutput,
    FraudDetectionOutput,
    RagUnderwritingOutput,
    RiskAssessmentOutput,
)


def run_decision_agent(
    fraud_detection: FraudDetectionOutput,
    risk_assessment: RiskAssessmentOutput,
    compliance: ComplianceOutput,
    rag_guidance: RagUnderwritingOutput,
) -> DecisionOutput:
    """Return approve, reject, or manual review based on explicit rules."""

    fraud_probability = fraud_detection.fraud_probability
    risk_score = risk_assessment.risk_score

    if fraud_probability >= 0.80 or risk_score >= 85:
        decision = "Reject"
        reason = "Claim exceeds rejection thresholds for fraud probability or risk score."
    elif (
        fraud_detection.fraud_risk == "High"
        or risk_assessment.risk_level == "High"
        or compliance.status == "Needs Review"
        or risk_score >= 55
    ):
        decision = "Manual Review"
        reason = "Claim requires human review due to elevated risk or compliance support gaps."
    else:
        decision = "Approve"
        reason = "Claim is within automated approval thresholds with source-backed guidance."

    return DecisionOutput(
        final_decision=decision,
        reasoning=reason,
        risk_score=risk_score,
        fraud_probability=fraud_probability,
        compliance_status=compliance.status,
        source_citations=rag_guidance.citations,
    )
