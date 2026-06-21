from pathlib import Path


def test_expected_project_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_directories = [
        "backend/app/data",
        "backend/app/ml",
        "backend/app/rag",
        "backend/app/agents",
        "backend/app/services",
        "backend/app/utils",
        "frontend",
        "data/raw_documents",
        "data/processed",
        "data/vector_store",
        "models",
        "notebooks",
        "tests",
    ]

    missing = [path for path in expected_directories if not (root / path).is_dir()]

    assert missing == []


def test_environment_template_contains_required_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    env_example = (root / ".env.example").read_text(encoding="utf-8")

    for key in ["OPENAI_API_KEY", "EMBEDDING_MODEL", "LLM_MODEL", "VECTOR_DB"]:
        assert f"{key}=" in env_example

