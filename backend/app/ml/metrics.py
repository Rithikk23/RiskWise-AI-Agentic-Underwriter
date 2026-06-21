"""Model metric helpers for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    """Serializable classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute binary fraud classification metrics."""

    roc_auc: float | None = None
    if y_probability is not None and len(set(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_probability))

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
    )
