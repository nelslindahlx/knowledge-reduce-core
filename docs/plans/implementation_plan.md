# Implementation Plan: Repository Hardening Alignment

Checklist bringing repository governance, zero-GitHub-minutes stratified local CI, Maker/Checker PR automation, hygiene guards, and GitHub branch protection from `KnowledgeReduce` to **KnowledgeReduce Core** (`knowledge-reduce-core`).

- `[x]` Create ephemeral feature branch `chore/repo-hardening` off `main`, remove stray root DB/watch artifacts, and install `ruff` dev tool.
- `[x]` Replace truncated `LICENSE` with full MIT License, add `CITATION.cff`, sync fallback `__version__ = "0.3.0"` in `knowledge_graph_pkg/__init__.py`, configure Ruff/coverage in `pyproject.toml`, update `.pre-commit-config.yaml` and `.gitignore`, and resolve test lint errors.
- `[x]` Scrub all machine-local workstation paths (absolute home paths and `file://` URLs) across `MODELREDUCE_MASTER_PLAN.md`, `MODELREDUCE_SESSIONS.md`, `deploy/`, `improvement_master_plan_phase_2.md`, `task.md`, and `walkthrough.md`.
- `[x]` Add `AGENTS.md`, byte-identical `.agents/AGENTS.md`, `.agents/rules/coding_method_guardrails.md`, `.agents/rules/release_and_pr_management.md`, `.agents/hooks.json`, `scripts/hooks/pre_pr_safety_gate.py`, and `scripts/github_pr_helper.py`, and update `scripts/run_suite.py`.
- `[x]` Add `.github/CODEOWNERS`, `.github/pull_request_template.md`, manual `workflow_dispatch`-only workflows (`ci.yml`, `pr-governance.yml`, `release.yml`, `skill-validation.yml`), and stratified `Makefile`.
- `[x]` Sync `CHANGELOG.md`, `CONTRIBUTING.md`, `README.md`, `lessons_learned.md`, `docs/plans/`, and add mechanical meta-tests in `tests/meta/test_governance_and_invariants.py`.
- `[x]` Verify `make local-ci`, `make smoke-pr`, PreToolUse hook deny/allow behavior, and zero path/pyc leaks.
- `[ ]` Configure GitHub `delete_branch_on_merge` and `main` branch protection, complete Maker/Checker PR audit + squash merge, and confirm `make audit-prs` exits `0`.
