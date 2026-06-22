"""Fraud detection agent wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.app.ml.fraud_predictor import predict_fraud
from backend.app.schemas import FraudDetectionOutput

FraudPredictor = Callable[[dict[str, Any]], dict[str, Any]]


def run_fraud_detection_agent(
    claim_record: dict[str, Any],
    *,
    predictor: FraudPredictor = predict_fraud,
) -> FraudDetectionOutput:
    """Score fraud risk for a claim record."""

    prediction = predictor(claim_record)
    return FraudDetectionOutput(
        fraud_probability=float(prediction["fraud_probability"]),
        fraud_risk=prediction["fraud_risk"],
        top_fraud_factors=list(prediction.get("top_fraud_factors", [])),
    )
