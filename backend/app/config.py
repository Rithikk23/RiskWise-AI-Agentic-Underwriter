"""Application configuration for local paths and model settings."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Path settings used by local services."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    claims_csv_path: Path = PROJECT_ROOT / "data" / "insurance_claims.csv"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    raw_documents_dir: Path = PROJECT_ROOT / "data" / "raw_documents"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    models_dir: Path = PROJECT_ROOT / "models"


settings = Settings()
