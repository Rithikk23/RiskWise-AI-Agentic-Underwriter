# Stage 1 PR Draft: Scaffold Python Project And Development Setup

## Title

Stage 1: Scaffold Python project and development setup

## Summary

This PR adds the initial Python project scaffold, dependency declarations, environment template, validation configuration, placeholder artifact directories, and a smoke test for expected repository structure.

## Changed Files

- `.env.example`
- `.gitignore`
- `requirements.txt`
- `backend/requirements.txt`
- `pyproject.toml`
- `backend/`
- `frontend/`
- `data/`
- `models/`
- `notebooks/`
- `tests/`
- `docs/development.md`
- `docs/pr/project-setup.md`

## Testing

- `python -m compileall backend`
- `python -m pytest`

## Risks

- Dependency versions are intentionally not pinned yet; lock files can be added once runtime compatibility is proven.
- Dataset, PDFs, vector indexes, logs, and model artifacts remain local-only.

## Acceptance Criteria

- Project package directories exist and are importable.
- Core dependencies are declared.
- `.env.example` includes required model and vector database settings.
- Pytest configuration works.
- No application behavior is implemented before its planned stage.

## Rollback Notes

Revert this scaffold commit. Local untracked datasets or generated artifacts are unaffected.

