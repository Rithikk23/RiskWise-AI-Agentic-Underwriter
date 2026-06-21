from pathlib import Path

import pandas as pd
import pytest

from backend.app.data.data_loader import (
    ClaimsDatasetNotFoundError,
    get_claim_by_id,
    get_sample_claims,
    load_claims_data,
)


def _write_claims_csv(path: Path) -> None:
    rows = [
        {
            "age": 45,
            "policy_state": "OH",
            "policy_annual_premium": 1406.91,
            "insured_occupation": "craft-repair",
            "incident_type": "Single Vehicle Collision",
            "collision_type": "?",
            "incident_severity": "Major Damage",
            "property_damage": "?",
            "bodily_injuries": 1,
            "witnesses": 2,
            "police_report_available": "NO",
            "total_claim_amount": 71610,
            "auto_make": "Saab",
            "auto_model": "92x",
            "auto_year": 2004,
            "fraud_reported": "Y",
        },
        {
            "age": None,
            "policy_state": "IN",
            "policy_annual_premium": 1197.22,
            "insured_occupation": "sales",
            "incident_type": "Vehicle Theft",
            "collision_type": "Unknown",
            "incident_severity": "Minor Damage",
            "property_damage": "NO",
            "bodily_injuries": 0,
            "witnesses": 0,
            "police_report_available": "YES",
            "total_claim_amount": 5070,
            "auto_make": "Dodge",
            "auto_model": "RAM",
            "auto_year": 2010,
            "fraud_reported": "N",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_claims_data_cleans_missing_values_and_target(tmp_path: Path) -> None:
    dataset_path = tmp_path / "insurance_claims.csv"
    _write_claims_csv(dataset_path)

    claims = load_claims_data(dataset_path)

    assert claims.loc[0, "claim_id"] == 0
    assert claims.loc[0, "collision_type"] == "Unknown"
    assert claims.loc[0, "property_damage"] == "Unknown"
    assert claims.loc[0, "fraud_reported"] == 1
    assert claims.loc[1, "fraud_reported"] == 0
    assert not claims.isna().any().any()


def test_get_claim_by_id_returns_cleaned_record(tmp_path: Path) -> None:
    dataset_path = tmp_path / "insurance_claims.csv"
    _write_claims_csv(dataset_path)

    claim = get_claim_by_id(1, dataset_path)

    assert claim["claim_id"] == 1
    assert claim["incident_type"] == "Vehicle Theft"
    assert claim["fraud_reported"] == 0


def test_get_claim_by_id_raises_for_unknown_claim(tmp_path: Path) -> None:
    dataset_path = tmp_path / "insurance_claims.csv"
    _write_claims_csv(dataset_path)

    with pytest.raises(KeyError, match="claim_id=99"):
        get_claim_by_id(99, dataset_path)


def test_get_sample_claims_respects_limit(tmp_path: Path) -> None:
    dataset_path = tmp_path / "insurance_claims.csv"
    _write_claims_csv(dataset_path)

    sample = get_sample_claims(1, dataset_path)

    assert len(sample) == 1
    assert sample[0]["claim_id"] == 0


def test_load_claims_data_preprocesses_for_model_features(tmp_path: Path) -> None:
    dataset_path = tmp_path / "insurance_claims.csv"
    _write_claims_csv(dataset_path)

    processed = load_claims_data(dataset_path, preprocess=True)

    assert processed["incident_type"].dtype.kind in {"i", "u"}
    assert processed["policy_state"].dtype.kind in {"i", "u"}
    assert processed["fraud_reported"].tolist() == [1, 0]
    assert processed["total_claim_amount"].between(0, 1).all()
    assert processed["claim_id"].tolist() == [0, 1]


def test_load_claims_data_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(ClaimsDatasetNotFoundError, match="Claims dataset not found"):
        load_claims_data(missing_path)
