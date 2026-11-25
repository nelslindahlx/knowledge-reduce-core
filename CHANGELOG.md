# Changelog

All notable changes to KnowledgeReduce Core are documented in this file.

## [Unreleased]

### Added
- **Repository Hardening & Maker/Checker PR Governance**:
  - Added `AGENTS.md` and byte-identical `.agents/AGENTS.md` plus `.agents/rules/coding_method_guardrails.md` and `.agents/rules/release_and_pr_management.md`.
  - Added Antigravity PreToolUse safety gate `.agents/hooks.json` backed by `scripts/hooks/pre_pr_safety_gate.py` (blocking direct pushes to `main` and running `tests/meta/` before PR creation/merge) and `scripts/github_pr_helper.py` targeting `nelslindahlx/knowledge-reduce-core`.
  - Added `.github/CODEOWNERS`, `.github/pull_request_template.md`, and manual `workflow_dispatch`-only `.github/workflows/ci.yml`, `.github/workflows/pr-governance.yml`, and `.github/workflows/release.yml` (zero automatic GitHub Actions minutes).
  - Added stratified local CI `Makefile` (`canary`, `skill`, `lint`, `smoke-pr`, `test`, `local-ci`, `audit-prs`, `clean`), `CITATION.cff`, `docs/plans/implementation_plan.md`, `docs/plans/latest_walkthrough.md`, and mechanical invariant checks in `tests/meta/test_governance_and_invariants.py`.

### Changed
- Converted `.github/workflows/skill-validation.yml` to manual `workflow_dispatch` only.
- Completed full MIT License text in `LICENSE`, synchronized fallback `__version__ = "0.3.0"` in `knowledge_graph_pkg/__init__.py`, added Ruff and coverage configuration in `pyproject.toml`, updated `.pre-commit-config.yaml` (`pre-commit-hooks` v5.0.0, `ruff-pre-commit` v0.8.0), expanded `.gitignore`, updated `CONTRIBUTING.md`, `README.md`, `lessons_learned.md`, and `scripts/run_suite.py`.
- Scrubbed machine-local filesystem paths from `MODELREDUCE_MASTER_PLAN.md`, `MODELREDUCE_SESSIONS.md`, `deploy/com.knowledgereduce.watcher.plist`, `deploy/knowledgereduce-watcher.service`, `improvement_master_plan_phase_2.md`, `task.md`, and `walkthrough.md`.

### Fixed
- Fixed missing `import json` (`F821`) in `tests/test_graph_compiler.py`, redundant `import pytest` (`F811`) in `tests/test_mcp_server.py`, unused variables (`F841`) in `tests/test_audit.py`, `tests/test_hardening.py`, `tests/test_ingest.py`, and `tests/test_watcher.py`, and invalid `<key>` argument tags in `deploy/com.knowledgereduce.watcher.plist`.

## [0.3.0] - 2026-07-18

### Added
- ModelReduce probing fleet (`Ollama`, `llama.cpp`, OpenAI-compatible, Gemini), cross-model consensus, gold-set eval gating, recursive weight crawling with logprob entropy profiling, and MLX/PyTorch SFT dataset compilation & training commands.
- Pluggable `BaseGraphStore` (`KuzuStore` and `Neo4jStore`), active graph reasoning (`auto_link_relations`, contradictions, transitive inference), incremental prefix-blocked entity resolution, configurable ontology distillation, hybrid PageRank + vector `GraphRAGRetriever`, `WatcherDaemon` with SQLite WAL tracking, `GraphTools`, and FastAPI + 3D WebGL dashboard (`mcp_server.py`).
- Hermes agent skill manifest (`.agents/skills/knowledge-reduce-core/SKILL.md`), `scripts/smoke_skill.py`, `scripts/run_suite.py`, and optional-dependency skip markers in `tests/conftest.py`.
