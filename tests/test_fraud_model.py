from pathlib import Path

import pandas as pd

from backend.app.data.data_loader import load_claims_data
from backend.app.ml.fraud_predictor import classify_fraud_risk, predict_fraud
from backend.app.ml.model_registry import load_model_artifact, save_model_artifact
from backend.app.ml.train_fraud_model import TARGET_COLUMN, train_fraud_model


def _write_training_csv(path: Path) -> None:
    rows = []
    base_rows = [
        (
            39,
            1200.0,
            0,
            "College",
            "sales",
            "Vehicle Theft",
            "Unknown",
            "Minor Damage",
            "Police",
            "NO",
            0,
            2,
            "YES",
            6000,
            500,
            1000,
            4500,
            "Dodge",
            "RAM",
            2011,
            "N",
        ),
        (
            45,
            1406.91,
            0,
            "High School",
            "craft-repair",
            "Single Vehicle Collision",
            "Front Collision",
            "Major Damage",
            "Police",
            "Unknown",
            1,
            0,
            "NO",
            71610,
            12000,
            18000,
            41610,
            "Saab",
            "92x",
            2004,
            "Y",
        ),
        (
            32,
            980.2,
            0,
            "Masters",
            "tech-support",
            "Parked Car",
            "Unknown",
            "Trivial Damage",
            "None",
            "NO",
            0,
            1,
            "YES",
            3500,
            0,
            500,
            3000,
            "Toyota",
            "Camry",
            2015,
            "N",
        ),
        (
            52,
            1690.4,
            1000000,
            "MD",
            "exec-managerial",
            "Multi-vehicle Collision",
            "Rear Collision",
            "Total Loss",
            "Ambulance",
            "YES",
            2,
            0,
            "NO",
            92000,
            25000,
            22000,
            45000,
            "BMW",
            "X5",
            2008,
            "Y",
        ),
        (
            28,
            760.5,
            0,
            "Associate",
            "adm-clerical",
            "Vehicle Theft",
            "Unknown",
            "Minor Damage",
            "Police",
            "NO",
            0,
            3,
            "YES",
            4800,
            0,
            800,
            4000,
            "Honda",
            "Civic",
            2018,
            "N",
        ),
        (
            61,
            2100.0,
            2000000,
            "JD",
            "priv-house-serv",
            "Single Vehicle Collision",
            "Side Collision",
            "Major Damage",
            "Fire",
            "YES",
            2,
            1,
            "NO",
            86000,
            20000,
            21000,
            45000,
            "Mercedes",
            "E400",
            2012,
            "Y",
        ),
        (
            36,
            1030.0,
            0,
            "College",
            "machine-op-inspct",
            "Parked Car",
            "Unknown",
            "Minor Damage",
            "None",
            "NO",
            0,
            2,
            "YES",
            7300,
            1000,
            1300,
            5000,
            "Ford",
            "Fusion",
            2016,
            "N",
        ),
        (
            48,
            1510.0,
            0,
            "Masters",
            "transport-moving",
            "Multi-vehicle Collision",
            "Front Collision",
            "Major Damage",
            "Police",
            "Unknown",
            1,
            0,
            "NO",
            65000,
            14000,
            16000,
            35000,
            "Audi",
            "A4",
            2009,
            "Y",
        ),
        (
            41,
            1120.0,
            0,
            "High School",
            "prof-specialty",
            "Vehicle Theft",
            "Unknown",
            "Minor Damage",
            "Police",
            "NO",
            0,
            2,
            "YES",
            9200,
            0,
            1200,
            8000,
            "Nissan",
            "Altima",
            2013,
            "N",
        ),
        (
            57,
            1840.0,
            750000,
            "PhD",
            "armed-forces",
            "Single Vehicle Collision",
            "Rear Collision",
            "Total Loss",
            "Ambulance",
            "YES",
            2,
            0,
            "NO",
            99000,
            30000,
            25000,
            44000,
            "Lexus",
            "RX",
            2007,
            "Y",
        ),
    ]
    columns = [
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
        "fraud_reported",
    ]
    for row in base_rows:
        rows.append(dict(zip(columns, row, strict=True)))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_train_fraud_model_returns_candidate_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "claims.csv"
    _write_training_csv(dataset_path)
    claims = load_claims_data(dataset_path)

    artifact = train_fraud_model(claims, random_state=7)

    assert artifact.target_column == TARGET_COLUMN
    assert artifact.selected_model_name in artifact.metrics_by_model
    assert "logistic_regression" in artifact.metrics_by_model
    assert "random_forest" in artifact.metrics_by_model
    assert "total_claim_amount" in artifact.feature_columns
    assert artifact.metrics_by_model[artifact.selected_model_name]["accuracy"] is not None


def test_model_registry_round_trips_artifact(tmp_path: Path) -> None:
    dataset_path = tmp_path / "claims.csv"
    model_path = tmp_path / "fraud_model.pkl"
    _write_training_csv(dataset_path)
    claims = load_claims_data(dataset_path)
    artifact = train_fraud_model(claims, random_state=7)

    saved_path = save_model_artifact(artifact, model_path)
    loaded = load_model_artifact(saved_path)

    assert saved_path == model_path
    assert loaded.selected_model_name == artifact.selected_model_name
    assert loaded.feature_columns == artifact.feature_columns


def test_predict_fraud_returns_required_shape(tmp_path: Path) -> None:
    dataset_path = tmp_path / "claims.csv"
    _write_training_csv(dataset_path)
    claims = load_claims_data(dataset_path)
    artifact = train_fraud_model(claims, random_state=7)
    claim = claims.iloc[1].to_dict()

    prediction = predict_fraud(claim, artifact=artifact)

    assert 0.0 <= prediction["fraud_probability"] <= 1.0
    assert prediction["fraud_risk"] in {"Low", "Medium", "High"}
    assert prediction["top_fraud_factors"]


def test_classify_fraud_risk_thresholds() -> None:
    assert classify_fraud_risk(0.30) == "Low"
    assert classify_fraud_risk(0.31) == "Medium"
    assert classify_fraud_risk(0.65) == "Medium"
    assert classify_fraud_risk(0.66) == "High"
