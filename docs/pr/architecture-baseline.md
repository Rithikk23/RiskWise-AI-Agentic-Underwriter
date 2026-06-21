# Stage 0 PR Draft: Add Architecture And Delivery Plan

## Title

Stage 0: Add architecture and delivery plan

## Summary

This PR establishes the architecture and safe staged delivery workflow before application coding begins. It documents the system components, runtime flow, assumptions, Git workflow, and review gates for the RiskWise-AI project.

## Changed Files

- `README.md`
- `docs/architecture.md`
- `docs/staged-implementation-plan.md`
- `.github/pull_request_template.md`
- `.gitignore`
- `docs/pr/architecture-baseline.md`

## Testing Notes

- Markdown reviewed for completeness.
- No application code is included in this stage.

## Risks

- Actual GitHub PR creation and branch protection cannot be completed until a remote repository is configured.
- Dataset, public PDFs, model artifacts, and vector indexes are intentionally excluded from Git by default.

## Acceptance Criteria

- Architecture summary and Mermaid diagram are present.
- All implementation stages include branch names, PR titles, tasks, files, tests, acceptance criteria, and rollback notes.
- No implementation code is introduced.
- Assumptions and remote/branch-protection limitations are documented.

## Rollback

Revert this documentation commit. No runtime state or generated artifacts are affected.
