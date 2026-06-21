# Staged Implementation Plan

Each stage must use a separate branch, create a pull request, pass validation, and stop for explicit approval before merge. Do not build on top of an unapproved branch.

## Stage 0: Architecture Baseline

- Branch: `architecture-baseline`
- PR title: `Stage 0: Add architecture and delivery plan`
- Implementation tasks: document architecture, assumptions, Git workflow, PR expectations, and stage boundaries.
- Files: `README.md`, `docs/architecture.md`, `docs/staged-implementation-plan.md`, `.github/pull_request_template.md`, `.gitignore`, `docs/pr/architecture-baseline.md`.
- Tests to run: `git status --short`, Markdown review.
- Acceptance criteria: no application code; architecture diagram present; all future stages have branch, PR title, tasks, files, tests, acceptance criteria, and rollback notes.
- Rollback notes: revert the architecture commit; no runtime behavior is affected.

## Stage 1: Project Setup

- Branch: `project-setup`
- PR title: `Stage 1: Scaffold Python project and development setup`
- Implementation tasks: create folder structure, Python package markers, dependency files, `.env.example`, pytest config, formatting/linting config, and local run instructions.
- Files: `backend/`, `backend/app/`, `frontend/`, `data/`, `models/`, `notebooks/`, `tests/`, `.env.example`, `requirements.txt`, `backend/requirements.txt`, `pyproject.toml`.
- Tests to run: `python -m compileall backend`, `python -m pytest`.
- Acceptance criteria: project imports cleanly; tests run with no application tests yet; environment variables are documented.
- Rollback notes: revert scaffold commit; no data or model artifacts should be removed.

## Stage 2: Dataset Loading

- Branch: `dataset-loading`
- PR title: `Stage 2: Add claims dataset loader and preprocessing`
- Implementation tasks: implement `load_claims_data()`, `get_claim_by_id()`, `get_sample_claims()`, replace `?`, map `fraud_reported` Y/N to 1/0, handle missing values, encode categoricals, normalize numerics.
- Files: `backend/app/data/data_loader.py`, `backend/app/data/preprocessing.py`, `backend/app/data/feature_engineering.py`, `backend/app/config.py`, `tests/test_data_loader.py`.
- Tests to run: `python -m pytest tests/test_data_loader.py`.
- Acceptance criteria: loader handles missing dataset with clear error; sample and ID lookup work; preprocessing produces model-ready data.
- Rollback notes: revert data modules and tests; local dataset remains untouched.

## Stage 3: Exploratory Data Analysis

- Branch: `eda`
- PR title: `Stage 3: Add fraud claims exploratory analysis notebook`
- Implementation tasks: create EDA notebook for fraud distribution, claim amount, severity, premium, vehicle year, state patterns, injuries, witnesses, and police report availability.
- Files: `notebooks/data_exploration.ipynb`, optional exported images under `docs/assets/eda/`, README links if images are added.
- Tests to run: `python -m pytest tests/test_data_loader.py`; notebook smoke execution if tooling is available.
- Acceptance criteria: notebook runs against local dataset; charts needed for README are generated or documented.
- Rollback notes: revert notebook and generated chart assets.

## Stage 4: Fraud Detection Model

- Branch: `fraud-model`
- PR title: `Stage 4: Train and serve fraud detection model`
- Implementation tasks: train LogisticRegression, RandomForestClassifier, and XGBoostClassifier; compute metrics; choose model; save artifact locally; implement prediction wrapper and factor explanations.
- Files: `backend/app/ml/train_fraud_model.py`, `backend/app/ml/fraud_predictor.py`, `backend/app/ml/model_registry.py`, `backend/app/ml/metrics.py`, `tests/test_fraud_model.py`, `notebooks/fraud_model_training.ipynb`.
- Tests to run: `python -m pytest tests/test_fraud_model.py`.
- Acceptance criteria: training command runs with local data; predictor returns fraud probability, risk label, and top factors; tests use small fixtures.
- Rollback notes: revert ML code/tests; remove generated local model artifact if created.

## Stage 5: RAG Ingestion Pipeline

- Branch: `rag-ingestion`
- PR title: `Stage 5: Add public document RAG ingestion pipeline`
- Implementation tasks: load PDFs from `data/raw_documents/`, extract text, clean text, chunk with size 800 and overlap 150, embed chunks, store FAISS index and metadata.
- Files: `backend/app/rag/ingest.py`, `backend/app/rag/chunking.py`, `backend/app/rag/embeddings.py`, `backend/app/rag/vector_store.py`, `backend/app/utils/pdf_loader.py`, `tests/test_rag.py`.
- Tests to run: `python -m pytest tests/test_rag.py`.
- Acceptance criteria: ingestion handles empty folder clearly; chunk metadata includes source, page, chunk ID, and document type; embedding calls are mockable.
- Rollback notes: revert RAG ingestion files/tests; delete generated local vector index if needed.

## Stage 6: RAG Retriever

- Branch: `rag-retriever`
- PR title: `Stage 6: Add source-cited RAG retriever`
- Implementation tasks: implement `retrieve_context(query, top_k=5)`, support guideline query types, return content, source, page, and score.
- Files: `backend/app/rag/retriever.py`, `backend/app/rag/vector_store.py`, `tests/test_rag.py`.
- Tests to run: `python -m pytest tests/test_rag.py`.
- Acceptance criteria: retriever returns stable structured results; missing vector store produces actionable error; tests use fake index data.
- Rollback notes: revert retriever changes without affecting ingestion code.

## Stage 7: Agent Implementations

- Branch: `agents`
- PR title: `Stage 7: Implement underwriting and fraud risk agents`
- Implementation tasks: implement customer profile, RAG underwriting, fraud detection, risk assessment, compliance, and decision agents with typed inputs and outputs.
- Files: `backend/app/agents/customer_profile_agent.py`, `backend/app/agents/rag_underwriting_agent.py`, `backend/app/agents/fraud_detection_agent.py`, `backend/app/agents/risk_assessment_agent.py`, `backend/app/agents/compliance_agent.py`, `backend/app/agents/decision_agent.py`, `backend/app/schemas.py`, `tests/test_agents.py`.
- Tests to run: `python -m pytest tests/test_agents.py`.
- Acceptance criteria: each agent is independently testable; decision rules match the plan; agent outputs include required fields.
- Rollback notes: revert agent modules and tests; no API behavior is exposed yet.

## Stage 8: LangGraph Workflow

- Branch: `langgraph-workflow`
- PR title: `Stage 8: Wire agents into LangGraph underwriting workflow`
- Implementation tasks: create sequential graph from customer profile through decision agent; expose `run_underwriting_workflow(claim_record, question)`; include agent trace.
- Files: `backend/app/agents/workflow.py`, `backend/app/services/underwriting_service.py`, `tests/test_agents.py`.
- Tests to run: `python -m pytest tests/test_agents.py`.
- Acceptance criteria: workflow returns final response schema; trace includes agent names and outputs; external calls remain mockable.
- Rollback notes: revert workflow/service changes while retaining standalone agents.

## Stage 9: FastAPI Backend

- Branch: `fastapi-backend`
- PR title: `Stage 9: Expose underwriting API endpoints`
- Implementation tasks: implement health, sample claims, claim lookup, underwriting, and retrieval endpoints with Pydantic schemas.
- Files: `backend/app/main.py`, `backend/app/schemas.py`, `backend/app/services/underwriting_service.py`, `tests/test_api.py`.
- Tests to run: `python -m pytest tests/test_api.py`.
- Acceptance criteria: all planned endpoints return documented shapes; invalid inputs return useful errors; tests use FastAPI test client.
- Rollback notes: revert API files/tests; underlying services remain available.

## Stage 10: Streamlit Frontend

- Branch: `streamlit-frontend`
- PR title: `Stage 10: Add Streamlit underwriting dashboard`
- Implementation tasks: build claim selector, profile panels, question input, workflow run action, decision card, fraud chart, risk gauge, compliance view, citations table, and agent trace.
- Files: `frontend/streamlit_app.py`, optional `frontend/README.md`, `README.md` screenshot placeholders.
- Tests to run: `python -m compileall frontend`; manual Streamlit smoke test.
- Acceptance criteria: UI can call local backend; empty/error states are handled; no private data is embedded.
- Rollback notes: revert frontend files; backend remains intact.

## Stage 11: Evaluation

- Branch: `evaluation`
- PR title: `Stage 11: Add model, retrieval, and workflow evaluation service`
- Implementation tasks: compute fraud metrics, decision consistency, retrieval relevance, citation coverage, and average latency through a command-line evaluation script.
- Files: `backend/app/services/evaluation_service.py`, `tests/test_fraud_model.py`, `tests/test_rag.py`, optional `docs/evaluation.md`.
- Tests to run: `python -m pytest tests/test_fraud_model.py tests/test_rag.py`.
- Acceptance criteria: evaluation command runs locally with clear missing-artifact errors; metrics are structured and documented.
- Rollback notes: revert evaluation service and docs.

## Stage 12: Observability

- Branch: `observability`
- PR title: `Stage 12: Add workflow logging and optional tracing hooks`
- Implementation tasks: log workflow ID, claim ID, agent name, summaries, latency, status, and errors; add optional LangSmith configuration without requiring it.
- Files: `backend/app/utils/logging.py`, `backend/app/agents/workflow.py`, `backend/app/config.py`, `.env.example`, `tests/test_agents.py`.
- Tests to run: `python -m pytest tests/test_agents.py`.
- Acceptance criteria: logs write to `logs/workflow.log` locally; tests verify log schema; tracing is disabled unless configured.
- Rollback notes: revert logging hooks; delete local logs if desired.

## Stage 13: Dockerization

- Branch: `dockerization`
- PR title: `Stage 13: Add Docker and Compose runtime`
- Implementation tasks: add backend and frontend Dockerfiles if needed, root `Dockerfile` only if it is the backend image, compose services, environment wiring, and health checks.
- Files: `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `.dockerignore`, `README.md`.
- Tests to run: `docker-compose config`; `docker-compose up --build` manual smoke test.
- Acceptance criteria: backend and frontend start locally through Compose; environment variables are documented; generated artifacts remain volume/local only.
- Rollback notes: revert Docker files; local images/containers may be removed manually.

## Stage 14: README And Portfolio Polish

- Branch: `readme-polish`
- PR title: `Stage 14: Complete GitHub-ready documentation`
- Implementation tasks: finish README with overview, business problem, dataset, architecture, agent workflow, RAG pipeline, fraud model, API endpoints, screenshots, install/run steps, examples, future enhancements, and resume alignment.
- Files: `README.md`, `docs/architecture.md`, `docs/assets/`, optional `docs/api.md`.
- Tests to run: full available test suite; link/path review.
- Acceptance criteria: README satisfies plan requirements; screenshots or placeholders are accurate; future enhancements are clearly marked out of scope.
- Rollback notes: revert documentation polish commit.

## Deferred Advanced Features

SHAP explainability, RAGAS evaluation, human-in-the-loop approval, PDF claim upload, PostgreSQL, role-based access control, Azure OpenAI, AKS/Kubernetes, GitHub Actions CI/CD, and Prometheus/Grafana are explicitly outside the MVP stages.
