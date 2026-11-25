## Summary
<!-- Concise summary of what changed and why. -->

## Linked Issue
<!-- Reference resolved or related GitHub issues, e.g. Closes #123 or Related to #123 -->

## Maker Checklist
- [ ] Changes live on a dedicated feature branch targeting `main` (never pushed directly to `main`).
- [ ] Local CI ran and passed on the latest commit (`make local-ci`: canary meta-gate, Hermes skill smoke/tests, `ruff check`, `ruff format --check` on governance/meta modules, and `pytest --cov=knowledge_graph_pkg --cov-fail-under=58`).
- [ ] Regression and negative tests cover new behavior/edge cases.
- [ ] Zero tracked `*.pyc` files, zero stray `.db` artifacts, and zero leaked machine-local paths.
- [ ] `CHANGELOG.md`, `lessons_learned.md`, and `docs/plans/` updated as appropriate.

## Checker Review
<!-- Completed independently by the Checker reviewer prior to merge. -->
- [ ] Diff reviewed (`git diff origin/main...HEAD`) for correctness, security invariants (workspace/path containment, JWT verification, read-only Cypher), `BaseGraphStore` / backend contract parity, and test rigor.
- [ ] Test assertions and edge cases audited (no relaxed/skipped assertions hiding bugs).
- [ ] Local CI verification outputs and timings recorded honestly.
- [ ] Merged cleanly via `scripts/github_pr_helper.py --merge --pr <N> --delete-branch` (or `gh pr merge --squash --delete-branch`) and audited via `--audit`.
