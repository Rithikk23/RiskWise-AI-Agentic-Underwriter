# RiskWise-AI: Multi-Agent Insurance Underwriting & Fraud Risk Platform

RiskWise-AI is a planned GitHub-ready Python project for simulating an insurance underwriting and claim-risk decision workflow. The MVP will combine a fraud detection model, public-document RAG, a LangGraph multi-agent workflow, FastAPI endpoints, and a Streamlit interface.

This repository is currently at **Stage 0: Architecture Baseline**. No application implementation has started yet. The architecture and staged delivery plan are committed first so each later implementation branch can be reviewed and approved independently.

## Architecture And Delivery Plan

- [Architecture](docs/architecture.md)
- [Staged implementation plan](docs/staged-implementation-plan.md)

## Current Assumptions

- Python 3.11 will be used unless a later stage proves a dependency requires otherwise.
- FAISS is the default local vector store.
- OpenAI is the default LLM and embedding provider through environment variables, with tests using mocks or deterministic fakes.
- The Kaggle `insurance_claims.csv` file and public insurance PDFs are supplied locally and are not committed by default.
- Generated artifacts such as trained models, vector indexes, logs, and notebooks outputs are excluded from Git unless a later PR explicitly adds small sample artifacts.
- GitHub branch protection and actual pull requests require a remote repository; this local repo currently has no remote configured.

## Git Workflow

Implementation work must follow the stage plan. Each stage gets its own branch, validation, pull request, and explicit approval before merge. Work must not continue on top of an unapproved branch.

