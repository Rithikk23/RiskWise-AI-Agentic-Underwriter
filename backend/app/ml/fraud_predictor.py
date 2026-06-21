"""Fraud prediction wrapper for trained model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.ml.model_registry import (
    DEFAULT_MODEL_PATH,
    FraudModelArtifact,
    load_model_artifact,
)

LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.65


def classify_fraud_risk(fraud_probability: float) -> str:
    """Classify a fraud probability into Low, Medium, or High."""

    if fraud_probability <= LOW_RISK_MAX:
        return "Low"
    if fraud_probability <= MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def explain_fraud_factors(claim_record: dict[str, Any], fraud_probability: float) -> list[str]:
    """Return deterministic high-level fraud factor explanations."""

    factors: list[str] = []

    total_claim_amount = _to_float(claim_record.get("total_claim_amount"))
    if total_claim_amount >= 60000:
        factors.append("High total claim amount")

    severity = str(claim_record.get("incident_severity", "")).lower()
    if "major" in severity or "total" in severity:
        factors.append("Severe incident")

    police_report = str(claim_record.get("police_report_available", "")).lower()
    if police_report in {"no", "unknown", "nan", ""}:
        factors.append("Police report unavailable")

    property_damage = str(claim_record.get("property_damage", "")).lower()
    if property_damage in {"yes", "unknown"}:
        factors.append("Property damage indicator")

    bodily_injuries = _to_float(claim_record.get("bodily_injuries"))
    if bodily_injuries >= 2:
        factors.append("Multiple bodily injuries")

    witnesses = _to_float(claim_record.get("witnesses"))
    if witnesses == 0:
        factors.append("No witnesses reported")

    if not factors:
        risk = classify_fraud_risk(fraud_probability)
        factors.append(f"{risk} model-estimated fraud risk")

    return factors[:3]


def predict_fraud(
    claim_record: dict[str, Any],
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    artifact: FraudModelArtifact | None = None,
) -> dict[str, Any]:
    """Predict fraud probability for one claim record."""

    model_artifact = artifact or load_model_artifact(model_path)
    features = _record_to_features(claim_record, model_artifact.feature_columns)
    probability = _predict_probability(model_artifact.model, features)

    return {
        "fraud_probability": round(probability, 4),
        "fraud_risk": classify_fraud_risk(probability),
        "top_fraud_factors": explain_fraud_factors(claim_record, probability),
    }


def _record_to_features(claim_record: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    row = {column: claim_record.get(column, "Unknown") for column in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def _predict_probability(model: Any, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return float(probabilities[0][1])

    prediction = model.predict(features)
    return float(prediction[0])


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
