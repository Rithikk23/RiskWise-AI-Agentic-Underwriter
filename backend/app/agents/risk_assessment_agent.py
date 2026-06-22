"""Risk assessment agent with deterministic scoring rules."""

from __future__ import annotations

from typing import Any

from backend.app.schemas import FraudDetectionOutput, RiskAssessmentOutput


def run_risk_assessment_agent(
    claim_record: dict[str, Any],
    fraud_detection: FraudDetectionOutput,
) -> RiskAssessmentOutput:
    """Compute a 0-100 risk score from claim and fraud factors."""

    score = int(round(fraud_detection.fraud_probability * 55))
    factors = [f"Fraud probability contributes {score} points"]

    claim_amount = _to_float(claim_record.get("total_claim_amount"))
    if claim_amount >= 75000:
        score += 20
        factors.append("Very high claim amount")
    elif claim_amount >= 50000:
        score += 12
        factors.append("High claim amount")

    severity = str(claim_record.get("incident_severity", "")).lower()
    if "total" in severity:
        score += 12
        factors.append("Total loss incident severity")
    elif "major" in severity:
        score += 8
        factors.append("Major incident severity")

    if str(claim_record.get("police_report_available", "")).lower() in {"no", "unknown"}:
        score += 8
        factors.append("Police report unavailable")

    if _to_float(claim_record.get("witnesses")) == 0:
        score += 5
        factors.append("No witnesses reported")

    if _to_float(claim_record.get("bodily_injuries")) >= 2:
        score += 5
        factors.append("Multiple bodily injuries")

    final_score = max(0, min(score, 100))
    return RiskAssessmentOutput(
        risk_score=final_score,
        risk_level=_risk_level(final_score),
        risk_factors=factors[:6],
    )


def _risk_level(score: int) -> str:
    if score < 35:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
