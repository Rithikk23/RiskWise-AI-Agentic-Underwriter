# Stage 7 PR Draft: Implement Underwriting and Fraud Risk Agents

## Title

Stage 7: Implement underwriting and fraud risk agents

## Summary

This PR adds independently testable agent modules for customer profile summarization, RAG underwriting guidance, fraud detection, deterministic risk scoring, compliance support checks, and final underwriting decisions. It also introduces shared Pydantic schemas for typed agent inputs and outputs.

## Changed Files

- `backend/app/schemas.py`
- `backend/app/agents/customer_profile_agent.py`
- `backend/app/agents/rag_underwriting_agent.py`
- `backend/app/agents/fraud_detection_agent.py`
- `backend/app/agents/risk_assessment_agent.py`
- `backend/app/agents/compliance_agent.py`
- `backend/app/agents/decision_agent.py`
- `backend/app/agents/__init__.py`
- `tests/test_agents.py`
- `docs/pr/agents.md`

## Testing

- `.venv/bin/python -m compileall backend`
- `.venv/bin/python -m pytest tests/test_project_structure.py tests/test_data_loader.py tests/test_fraud_model.py tests/test_rag.py tests/test_agents.py`
- `.venv/bin/python -m ruff check .`

## Risks

- Decision thresholds are deterministic MVP rules and may need calibration after end-to-end evaluation.
- RAG and fraud model dependencies are dependency-injected for tests; full orchestration is deferred to Stage 8.
- No API behavior is exposed in this stage.

## Acceptance Criteria

- Each agent can be tested independently.
- Agent outputs use typed schemas and include required fields.
- Fraud and RAG agents can use injected dependencies for tests.
- Decision agent returns `Approve`, `Reject`, or `Manual Review` according to explicit thresholds.
- Compliance agent checks whether source citations are present for source-backed decisions.

## Rollback Notes

Revert the agent modules, shared schemas, tests, and this PR draft. Existing dataset, ML, and RAG ingestion/retrieval stages remain unaffected.
