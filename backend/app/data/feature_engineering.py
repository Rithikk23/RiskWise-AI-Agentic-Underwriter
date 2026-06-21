"""Feature engineering helpers for claims data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMN = "fraud_reported"
IDENTIFIER_COLUMNS = {"claim_id"}


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata describing deterministic preprocessing transforms."""

    categorical_mappings: dict[str, dict[str, int]]
    normalized_columns: list[str]


def encode_categorical_columns(
    dataframe: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Encode object/category columns using stable integer mappings."""

    excluded = exclude_columns or set()
    encoded = dataframe.copy()
    mappings: dict[str, dict[str, int]] = {}

    categorical_columns = encoded.select_dtypes(include=["object", "category", "string"]).columns
    for column in categorical_columns:
        if column in excluded:
            continue

        values = encoded[column].astype("string").fillna("Unknown")
        categories = sorted(values.unique().tolist())
        mapping = {category: index for index, category in enumerate(categories)}
        encoded[column] = values.map(mapping).astype("int64")
        mappings[column] = mapping

    return encoded, mappings


def normalize_numeric_columns(
    dataframe: pd.DataFrame,
    *,
    include_columns: set[str] | None = None,
    exclude_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Min-max normalize numeric columns, leaving identifiers and targets untouched."""

    excluded = exclude_columns or set()
    normalized = dataframe.copy()
    normalized_columns: list[str] = []

    numeric_columns = normalized.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if include_columns is not None and column not in include_columns:
            continue
        if column in excluded:
            continue

        minimum = normalized[column].min()
        maximum = normalized[column].max()
        if pd.isna(minimum) or pd.isna(maximum):
            normalized[column] = 0.0
        elif maximum == minimum:
            normalized[column] = 0.0
        else:
            normalized[column] = (normalized[column] - minimum) / (maximum - minimum)
        normalized_columns.append(column)

    return normalized, normalized_columns


def prepare_model_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, FeatureMetadata]:
    """Return a model-ready dataframe with encoded categoricals and normalized numerics."""

    excluded = IDENTIFIER_COLUMNS | {TARGET_COLUMN}
    numeric_columns = set(dataframe.select_dtypes(include=["number"]).columns) - excluded
    encoded, mappings = encode_categorical_columns(dataframe, exclude_columns=excluded)
    normalized, normalized_columns = normalize_numeric_columns(
        encoded,
        include_columns=numeric_columns,
        exclude_columns=excluded,
    )

    metadata = FeatureMetadata(
        categorical_mappings=mappings,
        normalized_columns=normalized_columns,
    )
    return normalized, metadata
