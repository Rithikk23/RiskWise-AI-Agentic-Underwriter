"""Dataset loading helpers for insurance claim records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import settings
from backend.app.data.preprocessing import clean_claims_dataframe, preprocess_claims_dataframe


class ClaimsDatasetNotFoundError(FileNotFoundError):
    """Raised when the local insurance claims dataset is unavailable."""


def _resolve_dataset_path(dataset_path: str | Path | None = None) -> Path:
    return Path(dataset_path) if dataset_path is not None else settings.claims_csv_path


def load_claims_data(
    dataset_path: str | Path | None = None,
    *,
    preprocess: bool = False,
) -> pd.DataFrame:
    """Load the claims CSV and return cleaned or model-ready records."""

    path = _resolve_dataset_path(dataset_path)
    if not path.exists():
        raise ClaimsDatasetNotFoundError(
            f"Claims dataset not found at {path}. Add insurance_claims.csv under data/."
        )

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        return dataframe

    if preprocess:
        return preprocess_claims_dataframe(dataframe)
    return clean_claims_dataframe(dataframe)


def get_claim_by_id(
    claim_id: int,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return one cleaned claim record by stable claim_id."""

    claims = load_claims_data(dataset_path)
    matches = claims[claims["claim_id"] == claim_id]
    if matches.empty:
        raise KeyError(f"Claim with claim_id={claim_id} was not found.")

    return matches.iloc[0].to_dict()


def get_sample_claims(
    limit: int = 10,
    dataset_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return a small list of cleaned claim records for UI/API sampling."""

    if limit < 1:
        return []

    claims = load_claims_data(dataset_path)
    return claims.head(limit).to_dict(orient="records")
