# Stage 4 PR Draft: Train And Serve Fraud Detection Model

## Title

Stage 4: Train and serve fraud detection model

## Summary

This PR adds the fraud model training pipeline, model registry, metric helpers, prediction wrapper, model-training notebook, and focused tests. The trainer compares Logistic Regression, Random Forest, and XGBoost when available, then saves the selected preprocessing-and-model pipeline as a local ignored artifact.

## Changed Files

- `backend/app/ml/train_fraud_model.py`
- `backend/app/ml/fraud_predictor.py`
- `backend/app/ml/model_registry.py`
- `backend/app/ml/metrics.py`
- `tests/test_fraud_model.py`
- `notebooks/fraud_model_training.ipynb`
- `docs/pr/fraud-model.md`

## Testing

- `.venv/bin/python -m compileall backend`
- `.venv/bin/python -m pytest tests/test_project_structure.py tests/test_data_loader.py tests/test_fraud_model.py`
- `.venv/bin/python -m ruff check .`
- `.venv/bin/python backend/app/ml/train_fraud_model.py`

## Risks

- `models/fraud_model.pkl` is generated locally and ignored by Git.
- Model quality depends on the local Kaggle CSV and should be revisited during evaluation stages.
- XGBoost is optional at runtime; the trainer still supports Logistic Regression and Random Forest.

## Acceptance Criteria

- Training command runs against the local dataset.
- Candidate model metrics include accuracy, precision, recall, F1, and ROC-AUC where available.
- Predictor returns fraud probability, risk label, and top fraud factors.
- Tests use fixture data and do not require the raw dataset.

## Rollback Notes

Revert the ML modules, notebook, tests, and this PR draft. Delete local `models/fraud_model.pkl` if desired.
