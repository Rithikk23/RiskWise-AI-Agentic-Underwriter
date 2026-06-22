"""Customer profile agent for readable claim summaries."""

from __future__ import annotations

from typing import Any

from backend.app.schemas import CustomerProfileOutput


def build_customer_profile(claim_record: dict[str, Any]) -> CustomerProfileOutput:
    """Summarize customer, policy, incident, and claim details."""

    claim_id = _optional_int(claim_record.get("claim_id"))
    age = claim_record.get("age", "Unknown")
    relationship = claim_record.get("insured_relationship", "Unknown")
    occupation = claim_record.get("insured_occupation", "Unknown")
    policy_state = claim_record.get("policy_state", "Unknown")
    deductible = claim_record.get("policy_deductable", "Unknown")
    premium = claim_record.get("policy_annual_premium", "Unknown")
    severity = claim_record.get("incident_severity", "Unknown")
    incident_type = claim_record.get("incident_type", "Unknown")
    incident_state = claim_record.get("incident_state", "Unknown")
    claim_amount = _money(claim_record.get("total_claim_amount"))
    flags = _profile_flags(claim_record)

    return CustomerProfileOutput(
        claim_id=claim_id,
        customer_summary=(
            f"Claim {claim_id if claim_id is not None else 'Unknown'} involves a "
            f"{age}-year-old insured listed as {relationship} with occupation {occupation}."
        ),
        policy_summary=(
            f"Policy state is {policy_state}, deductible is {deductible}, "
            f"and annual premium is {premium}."
        ),
        incident_summary=(
            f"Incident type is {incident_type} in {incident_state} with severity {severity}."
        ),
        claim_summary=f"Total claimed amount is {claim_amount}.",
        profile_flags=flags,
    )


def _profile_flags(claim_record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if _to_float(claim_record.get("total_claim_amount")) >= 60000:
        flags.append("High total claim amount")
    if str(claim_record.get("incident_severity", "")).lower() in {
        "major damage",
        "total loss",
    }:
        flags.append("Severe incident")
    if str(claim_record.get("police_report_available", "")).lower() in {"no", "unknown"}:
        flags.append("Police report not confirmed")
    if _to_float(claim_record.get("witnesses")) == 0:
        flags.append("No witnesses reported")
    return flags


def _money(value: object) -> str:
    amount = _to_float(value)
    if amount == 0:
        return "$0"
    return f"${amount:,.0f}"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
