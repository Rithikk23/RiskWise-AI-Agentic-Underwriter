from backend.app.agents import (
    build_customer_profile,
    run_compliance_agent,
    run_decision_agent,
    run_fraud_detection_agent,
    run_rag_underwriting_agent,
    run_risk_assessment_agent,
)
from backend.app.schemas import FraudDetectionOutput, RagUnderwritingOutput, RiskAssessmentOutput


def _claim_record() -> dict[str, object]:
    return {
        "claim_id": 7,
        "age": 45,
        "policy_state": "OH",
        "policy_deductable": 1000,
        "policy_annual_premium": 1406.91,
        "insured_relationship": "husband",
        "insured_occupation": "craft-repair",
        "incident_type": "Single Vehicle Collision",
        "incident_state": "OH",
        "incident_severity": "Major Damage",
        "police_report_available": "NO",
        "bodily_injuries": 2,
        "witnesses": 0,
        "total_claim_amount": 71610,
    }


def test_customer_profile_agent_summarizes_required_sections() -> None:
    profile = build_customer_profile(_claim_record())

    assert profile.claim_id == 7
    assert profile.agent_name == "customer_profile_agent"
    assert "45-year-old" in profile.customer_summary
    assert "Policy state is OH" in profile.policy_summary
    assert "Single Vehicle Collision" in profile.incident_summary
    assert "High total claim amount" in profile.profile_flags


def test_fraud_detection_agent_uses_injected_predictor() -> None:
    output = run_fraud_detection_agent(
        _claim_record(),
        predictor=lambda _claim: {
            "fraud_probability": 0.72,
            "fraud_risk": "High",
            "top_fraud_factors": ["High total claim amount"],
        },
    )

    assert output.fraud_probability == 0.72
    assert output.fraud_risk == "High"
    assert output.top_fraud_factors == ["High total claim amount"]


def test_rag_underwriting_agent_returns_citations_from_retriever() -> None:
    output = run_rag_underwriting_agent(
        "What claim handling rules apply?",
        retriever=lambda *_args, **_kwargs: [
            {
                "content": "Claims should receive timely and fair review.",
                "source": "claims.pdf",
                "page": 3,
                "score": 0.91,
            }
        ],
    )

    assert output.query_type == "claim_handling"
    assert output.citations[0].source == "claims.pdf"
    assert output.citations[0].page == 3
    assert "Retrieved 1 public guidance" in output.guidance_summary


def test_risk_assessment_agent_scores_high_risk_claim() -> None:
    fraud = FraudDetectionOutput(
        fraud_probability=0.72,
        fraud_risk="High",
        top_fraud_factors=["High total claim amount"],
    )

    risk = run_risk_assessment_agent(_claim_record(), fraud)

    assert risk.risk_score >= 70
    assert risk.risk_level == "High"
    assert "Police report unavailable" in risk.risk_factors


def test_compliance_agent_requires_citations_for_high_risk() -> None:
    guidance = RagUnderwritingOutput(
        query="claim handling",
        query_type="claim_handling",
        guidance_summary="No public guidance excerpts were retrieved.",
        citations=[],
    )
    risk = RiskAssessmentOutput(risk_score=80, risk_level="High", risk_factors=[])

    compliance = run_compliance_agent(guidance, risk)

    assert compliance.status == "Needs Review"
    assert compliance.is_supported is False
    assert "No public source citations retrieved" in compliance.issues


def test_decision_agent_rejects_extreme_risk_and_approves_low_risk() -> None:
    high_fraud = FraudDetectionOutput(
        fraud_probability=0.86,
        fraud_risk="High",
        top_fraud_factors=["Severe incident"],
    )
    high_risk = RiskAssessmentOutput(risk_score=90, risk_level="High", risk_factors=[])
    guidance = run_rag_underwriting_agent(
        "claim handling",
        retriever=lambda *_args, **_kwargs: [
            {
                "content": "Fair review is required.",
                "source": "claims.pdf",
                "page": 1,
                "score": 0.9,
            },
            {
                "content": "Documentation should be retained.",
                "source": "claims.pdf",
                "page": 2,
                "score": 0.8,
            },
        ],
    )
    compliant = run_compliance_agent(guidance, high_risk)

    rejected = run_decision_agent(high_fraud, high_risk, compliant, guidance)

    assert rejected.final_decision == "Reject"

    low_fraud = FraudDetectionOutput(
        fraud_probability=0.12,
        fraud_risk="Low",
        top_fraud_factors=["Low model-estimated fraud risk"],
    )
    low_risk = RiskAssessmentOutput(risk_score=18, risk_level="Low", risk_factors=[])
    low_compliance = run_compliance_agent(guidance, low_risk)

    approved = run_decision_agent(low_fraud, low_risk, low_compliance, guidance)

    assert approved.final_decision == "Approve"
    assert approved.source_citations
