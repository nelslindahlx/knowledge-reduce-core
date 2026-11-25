# Changelog

All notable changes to KnowledgeReduce Core are documented in this file.

## [Unreleased]

### Added
- **Phase 1 Security, Graph Store Parity, MCP Dashboard & Watcher Hardening**:
  - Added shared `normalize_fact_record`, `_block_id`, `_s`, and `RELIABILITY_RANK` in `knowledge_graph_pkg/graph_store_base.py`, unifying canonical/legacy schema mapping across `KuzuStore` and `Neo4jStore`.
  - Added `Neo4jStore.with_database()`, explicit `NEO4J_PASSWORD`/argument validation, `bolt+s://`/`bolt+ssc://` routing, and credential-stripped URI construction in `knowledge_graph_pkg/graph_store_factory.py`.
  - Hardened `GraphTools.graph_query` against procedure calls (`CALL`), `FOREACH`/`EXPORT`/`IMPORT`, embedded semicolons, and comments while allowing quoted keywords and enforcing Python slice caps.
  - Added real end-to-end integration and security regression tests in `tests/test_watcher.py`, `tests/test_mcp_server.py`, `tests/test_neo4j_store.py`, `tests/test_graph_tool.py`, and `tests/test_reasoning.py`.
  - Logged full 6-phase audit & remediation roadmap in `docs/plans/master_plan_v0_4_0.md`.
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
- Fixed `mcp_server.py` security and runtime issues: removed raw secret bearer bypass in `verify_token`, added JWT verification on `/api/notify_update`, `WS /ws`, and stdlib `make_handler`, partitioned WebSockets by workspace, restricted `x-workspace-id` via allowlist + path containment (`KuzuStore.path`), added LRU workspace store eviction, restricted CORS defaults, eliminated double `retrieve()` in `/api/rag_retrieve`, and fixed `DASHBOARD_HTML` XSS (`escapeHtml` + `data-block-id`), `edgesDataset` / `token` `ReferenceError`s, and WebGL renderer reuse.
- Fixed dead `WatcherDaemon` graph pipeline by adding `"fact_records"` to `factory.batch_drop`, preserving Neo4j URIs, adding `on_moved`, syncing graph operations before marking `PROCESSED`, and writing `ontology_summary.json` atomically.
- Fixed `BaseGraphStore.auto_link_relations` (ignore empty subjects/objects, preserve existing weights), `find_contradictions` (require core predicate overlap), `find_transitive_inferences` (case-insensitive, exclude `UNVERIFIED` and self-loops), and `validate_and_reconcile` (rank by reliability tier + quality).
- Fixed missing `import json` (`F821`) in `tests/test_graph_compiler.py`, redundant `import pytest` (`F811`) in `tests/test_mcp_server.py`, unused variables (`F841`) in `tests/test_audit.py`, `tests/test_hardening.py`, `tests/test_ingest.py`, and `tests/test_watcher.py`, and invalid `<key>` argument tags in `deploy/com.knowledgereduce.watcher.plist`.

## [0.3.0] - 2026-07-18

### Added
- ModelReduce probing fleet (`Ollama`, `llama.cpp`, OpenAI-compatible, Gemini), cross-model consensus, gold-set eval gating, recursive weight crawling with logprob entropy profiling, and MLX/PyTorch SFT dataset compilation & training commands.
- Pluggable `BaseGraphStore` (`KuzuStore` and `Neo4jStore`), active graph reasoning (`auto_link_relations`, contradictions, transitive inference), incremental prefix-blocked entity resolution, configurable ontology distillation, hybrid PageRank + vector `GraphRAGRetriever`, `WatcherDaemon` with SQLite WAL tracking, `GraphTools`, and FastAPI + 3D WebGL dashboard (`mcp_server.py`).
- Hermes agent skill manifest (`.agents/skills/knowledge-reduce-core/SKILL.md`), `scripts/smoke_skill.py`, `scripts/run_suite.py`, and optional-dependency skip markers in `tests/conftest.py`.
