# Development

## Local Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` for local secrets. Do not commit `.env`.

## Validation

Stage 1 validation commands:

```bash
python -m compileall backend
python -m pytest
```

Later stages will add service-specific test commands as implementation code lands.

## Local Artifacts

The following paths are intentionally present but mostly ignored by Git:

- `data/insurance_claims.csv`
- `data/raw_documents/`
- `data/processed/`
- `data/vector_store/`
- `models/`
- `logs/`

