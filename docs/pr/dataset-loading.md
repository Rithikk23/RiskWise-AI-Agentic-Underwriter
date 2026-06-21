# Stage 2 PR Draft: Add Claims Dataset Loader And Preprocessing

## Title

Stage 2: Add claims dataset loader and preprocessing

## Summary

This PR adds the claims dataset loading layer, preprocessing helpers, feature engineering utilities, path configuration, and focused tests. The loader returns cleaned human-readable records by default and model-ready encoded/normalized records when `preprocess=True`.

## Changed Files

- `backend/app/config.py`
- `backend/app/data/data_loader.py`
- `backend/app/data/preprocessing.py`
- `backend/app/data/feature_engineering.py`
- `tests/test_data_loader.py`
- `docs/pr/dataset-loading.md`

## Testing

- `.venv/bin/python -m compileall backend`
- `.venv/bin/python -m pytest tests/test_project_structure.py tests/test_data_loader.py`
- `.venv/bin/python -m ruff check .`

## Risks

- The real Kaggle dataset is not committed and was not present locally during validation; tests use representative fixture rows.
- Categorical encoding is deterministic within each loaded dataset, but Stage 4 may replace it with a persisted model pipeline if needed.

## Acceptance Criteria

- Missing `data/insurance_claims.csv` raises a clear error.
- `load_claims_data()` cleans `?`, missing values, and `fraud_reported`.
- `get_claim_by_id()` returns a cleaned record or raises a clear lookup error.
- `get_sample_claims()` respects the requested limit.
- `load_claims_data(preprocess=True)` returns model-ready encoded and normalized data.

## Rollback Notes

Revert the data modules, config file, tests, and this PR draft. Local datasets remain untouched.
