"""Cleaning utilities for the insurance claims dataset."""

from __future__ import annotations

import pandas as pd

from backend.app.data.feature_engineering import prepare_model_features

MISSING_TEXT_VALUE = "Unknown"
TARGET_COLUMN = "fraud_reported"


def replace_unknown_markers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace Kaggle missing-value markers with a consistent text value."""

    cleaned = dataframe.copy()
    cleaned = cleaned.replace("?", MISSING_TEXT_VALUE)

    text_columns = cleaned.select_dtypes(include=["object", "category", "string"]).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace("", MISSING_TEXT_VALUE)
        cleaned[column] = cleaned[column].fillna(MISSING_TEXT_VALUE)

    return cleaned


def convert_fraud_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert fraud_reported values from Y/N to 1/0 when present."""

    converted = dataframe.copy()
    if TARGET_COLUMN not in converted.columns:
        return converted

    mapping = {"Y": 1, "N": 0, "YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0}
    target_values = converted[TARGET_COLUMN]

    if pd.api.types.is_numeric_dtype(target_values):
        converted[TARGET_COLUMN] = target_values.fillna(0).astype("int64")
        return converted

    normalized = target_values.astype("string").str.strip().str.upper()
    converted[TARGET_COLUMN] = normalized.map(mapping).fillna(0).astype("int64")
    return converted


def fill_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for text and numeric columns without dropping rows."""

    filled = dataframe.copy()

    text_columns = filled.select_dtypes(include=["object", "category", "string"]).columns
    for column in text_columns:
        filled[column] = filled[column].fillna(MISSING_TEXT_VALUE)

    numeric_columns = filled.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if filled[column].isna().any():
            median = filled[column].median()
            fill_value = 0 if pd.isna(median) else median
            filled[column] = filled[column].fillna(fill_value)

    return filled


def ensure_claim_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure each row has a stable integer claim_id."""

    with_ids = dataframe.copy()
    if "claim_id" not in with_ids.columns:
        with_ids.insert(0, "claim_id", range(len(with_ids)))
        return with_ids

    with_ids["claim_id"] = pd.to_numeric(with_ids["claim_id"], errors="coerce")
    if with_ids["claim_id"].isna().any():
        fallback_ids = pd.Series(range(len(with_ids)), index=with_ids.index)
        with_ids["claim_id"] = with_ids["claim_id"].fillna(fallback_ids)
    with_ids["claim_id"] = with_ids["claim_id"].astype("int64")
    return with_ids


def clean_claims_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean the claims dataset while preserving readable categorical values."""

    cleaned = ensure_claim_id(dataframe)
    cleaned = replace_unknown_markers(cleaned)
    cleaned = convert_fraud_target(cleaned)
    cleaned = fill_missing_values(cleaned)
    return cleaned


def preprocess_claims_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean, encode, and normalize claims data for downstream model training."""

    cleaned = clean_claims_dataframe(dataframe)
    processed, _metadata = prepare_model_features(cleaned)
    return processed
