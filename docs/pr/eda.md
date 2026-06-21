# Stage 3 PR Draft: Add Fraud Claims Exploratory Analysis Notebook

## Title

Stage 3: Add fraud claims exploratory analysis notebook

## Summary

This PR adds an executable EDA notebook for the local `insurance_claims.csv` dataset and static chart assets for later README usage. The analysis covers fraud distribution, claim amount distribution, incident severity, policy premium, vehicle year, state patterns, bodily injuries, witnesses, and police report availability.

## Changed Files

- `notebooks/data_exploration.ipynb`
- `docs/assets/eda/README.md`
- `docs/assets/eda/*.png`
- `docs/pr/eda.md`

## Testing

- `.venv/bin/python -m compileall backend`
- `.venv/bin/python -m pytest tests/test_project_structure.py tests/test_data_loader.py`
- `.venv/bin/python -m ruff check .`
- Notebook JSON parsed successfully.
- Chart assets regenerated from local CSV.

## Risks

- The notebook depends on the local ignored dataset at `data/insurance_claims.csv`.
- Static chart values reflect the local CSV version supplied during this stage.

## Acceptance Criteria

- Notebook loads the project data loader.
- Notebook includes all planned EDA views.
- Chart assets exist under `docs/assets/eda/`.
- No raw dataset is committed.

## Rollback Notes

Revert the notebook, generated chart assets, and this PR draft. The ignored local CSV remains untouched.
