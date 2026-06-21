"""Local model artifact registry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from backend.app.config import settings

DEFAULT_MODEL_PATH = settings.models_dir / "fraud_model.pkl"


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when a model artifact is expected but unavailable."""


@dataclass(frozen=True)
class FraudModelArtifact:
    """Persisted fraud model bundle."""

    model: Any
    feature_columns: list[str]
    selected_model_name: str
    metrics_by_model: dict[str, dict[str, float | None]]
    target_column: str
    trained_at: str


def save_model_artifact(
    artifact: FraudModelArtifact,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    """Save a fraud model artifact to disk."""

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> FraudModelArtifact:
    """Load a fraud model artifact from disk."""

    path = Path(model_path)
    if not path.exists():
        raise ModelArtifactNotFoundError(
            f"Fraud model artifact not found at {path}. Run backend/app/ml/train_fraud_model.py."
        )
    return joblib.load(path)
