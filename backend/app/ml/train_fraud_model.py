"""Train and persist fraud detection models."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from backend.app.data.data_loader import load_claims_data  # noqa: E402
from backend.app.ml.metrics import compute_classification_metrics  # noqa: E402
from backend.app.ml.model_registry import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FraudModelArtifact,
    save_model_artifact,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - depends on optional native xgboost runtime
    XGBClassifier = None


TARGET_COLUMN = "fraud_reported"
MODEL_FEATURE_COLUMNS = [
    "age",
    "policy_annual_premium",
    "umbrella_limit",
    "insured_education_level",
    "insured_occupation",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "property_damage",
    "bodily_injuries",
    "witnesses",
    "police_report_available",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_make",
    "auto_model",
    "auto_year",
]


def train_and_save_model(
    *,
    dataset_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    random_state: int = 42,
) -> FraudModelArtifact:
    """Train candidate fraud models and save the best artifact."""

    claims = load_claims_data(dataset_path)
    artifact = train_fraud_model(claims, random_state=random_state)
    save_model_artifact(artifact, model_path)
    return artifact


def train_fraud_model(
    claims: pd.DataFrame,
    *,
    random_state: int = 42,
) -> FraudModelArtifact:
    """Train fraud model candidates and return the selected artifact."""

    feature_columns = [column for column in MODEL_FEATURE_COLUMNS if column in claims.columns]
    if TARGET_COLUMN not in claims.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing.")
    if not feature_columns:
        raise ValueError("No configured fraud model feature columns are present.")

    X = claims[feature_columns]
    y = claims[TARGET_COLUMN].astype(int)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    candidates = build_candidate_models(X_train, random_state=random_state)
    trained_models: dict[str, Pipeline] = {}
    metrics_by_model: dict[str, dict[str, float | None]] = {}

    for model_name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = _predict_probabilities(model, X_test)
        metrics = compute_classification_metrics(
            y_test.to_numpy(),
            predictions,
            probabilities,
        )
        trained_models[model_name] = model
        metrics_by_model[model_name] = metrics.to_dict()

    selected_model_name = select_best_model(metrics_by_model)
    return FraudModelArtifact(
        model=trained_models[selected_model_name],
        feature_columns=feature_columns,
        selected_model_name=selected_model_name,
        metrics_by_model=metrics_by_model,
        target_column=TARGET_COLUMN,
        trained_at=datetime.now(UTC).isoformat(),
    )


def build_candidate_models(
    X: pd.DataFrame,
    *,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Build candidate sklearn pipelines for available model families."""

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    candidates: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
    }

    if XGBClassifier is not None:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        for name, estimator in candidates.items()
    }


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build preprocessing for mixed numeric and categorical claim fields."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def select_best_model(metrics_by_model: dict[str, dict[str, float | None]]) -> str:
    """Select the best model by ROC-AUC, then F1, then accuracy."""

    def score(item: tuple[str, dict[str, float | None]]) -> tuple[float, float, float]:
        metrics = item[1]
        return (
            float(metrics.get("roc_auc") or 0.0),
            float(metrics.get("f1") or 0.0),
            float(metrics.get("accuracy") or 0.0),
        )

    return max(metrics_by_model.items(), key=score)[0]


def _predict_probabilities(model: Pipeline, X_test: pd.DataFrame) -> Any:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return None


def main() -> None:
    artifact = train_and_save_model()
    print(f"Selected model: {artifact.selected_model_name}")
    for model_name, metrics in artifact.metrics_by_model.items():
        print(f"{model_name}: {metrics}")
    print(f"Saved model: {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
