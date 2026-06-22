"""Shared typed schemas for RiskWise-AI agent outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["Low", "Medium", "High"]
ComplianceStatus = Literal["Compliant", "Needs Review"]
UnderwritingDecision = Literal["Approve", "Reject", "Manual Review"]


class Citation(BaseModel):
    """Source citation returned from the RAG layer."""

    source: str
    page: int | None = None
    score: float | None = None
    excerpt: str


class AgentInput(BaseModel):
    """Shared input shape for agent-level tests and workflow composition."""

    claim_record: dict[str, Any]
    question: str = "Assess this claim for underwriting and fraud risk."


class CustomerProfileOutput(BaseModel):
    """Customer, policy, incident, and claim summary."""

    agent_name: str = "customer_profile_agent"
    claim_id: int | None = None
    customer_summary: str
    policy_summary: str
    incident_summary: str
    claim_summary: str
    profile_flags: list[str] = Field(default_factory=list)


class FraudDetectionOutput(BaseModel):
    """Fraud model score and explanation."""

    agent_name: str = "fraud_detection_agent"
    fraud_probability: float
    fraud_risk: RiskLevel
    top_fraud_factors: list[str] = Field(default_factory=list)


class RagUnderwritingOutput(BaseModel):
    """Retrieved public guidance with source citations."""

    agent_name: str = "rag_underwriting_agent"
    query: str
    query_type: str
    guidance_summary: str
    citations: list[Citation] = Field(default_factory=list)


class RiskAssessmentOutput(BaseModel):
    """Deterministic claim risk score and factors."""

    agent_name: str = "risk_assessment_agent"
    risk_score: int
    risk_level: RiskLevel
    risk_factors: list[str] = Field(default_factory=list)


class ComplianceOutput(BaseModel):
    """Compliance support check against retrieved public sources."""

    agent_name: str = "compliance_agent"
    status: ComplianceStatus
    is_supported: bool
    issues: list[str] = Field(default_factory=list)
    citations_checked: int = 0


class DecisionOutput(BaseModel):
    """Final underwriting recommendation."""

    agent_name: str = "decision_agent"
    final_decision: UnderwritingDecision
    reasoning: str
    risk_score: int
    fraud_probability: float
    compliance_status: ComplianceStatus
    source_citations: list[Citation] = Field(default_factory=list)
