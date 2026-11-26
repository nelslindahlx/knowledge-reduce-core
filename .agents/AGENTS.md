# Project Rules & Agent Governance

This file defines repository rules and conventions for **KnowledgeReduce Core** (`knowledgereduce` / `knowledge_graph_pkg`), aligned with zero-GitHub-Actions-minutes local CI and the two-part Maker/Checker framework. Both `AGENTS.md` and `.agents/AGENTS.md` must stay byte-identical and under `24,000` bytes.

---

## 1. Zero GitHub Actions Minutes & Stratified Local CI

- **Manual-Only Workflows (`workflow_dispatch`)**: All GitHub Actions workflows in `.github/workflows/` MUST trigger strictly via `on: workflow_dispatch:` (no automatic `push` or `pull_request` triggers) so routine PRs and merges consume **zero** GitHub Actions minutes.
- **Stratified Local CI Commands (`Makefile`)**:
  - **Canary Meta Gate (`< 10s`)**: `make canary` (`uv run --no-sync python -W error::Warning -c "import knowledge_graph_pkg"` + `pytest tests/meta/ -q`).
  - **Hermes Skill Gate**: `make skill` (`python scripts/smoke_skill.py` + `pytest tests/test_skill.py -q`).
  - **Lint & Format**: `make lint` (`ruff check knowledge_graph_pkg tests scripts benchmarks` plus strict `ruff check` and `ruff format --check` on governance scripts, `tests/meta`, `benchmarks`, and `tests/golden`).
  - **Golden Regression Gate**: `make golden` (offline `civichonors_book_benchmark.py` and `multisource_factory_benchmark.py` with `--strict-regression`) and `tests/golden/`.
  - **PR Smoke Tier**: `make smoke-pr` (`canary` + `skill` + `lint` + Stage 1 & Stage 2 via `scripts/run_suite.py`).
  - **Full Local CI (Mandatory Before PR Merge)**: `make local-ci` (`canary`, `skill`, `lint`, `golden`, and `pytest --cov=knowledge_graph_pkg --cov-report=term-missing --cov-fail-under=75`).

---

## 2. Two-Part Maker / Checker Pull Request Framework & Pre-Flight Guard

All modifications MUST flow through a two-part **Maker / Checker** GitHub Pull Request lifecycle:

- **Pre-Flight Branch Guard**: Never commit or push directly to `main`. Before editing any files, run `git branch --show-current`; if on `main`, switch immediately to an ephemeral branch (`feat/...`, `fix/...`, `chore/...`, `docs/...`, `refactor/...`). Once a branch is merged, never reuse its name.
- **Direct Push Ban & Mechanical Pre-PR Hook**: `.agents/hooks.json` (`scripts/hooks/pre_pr_safety_gate.py`) hard-denies `git push origin main` and automatically executes `pytest tests/meta/ -q` prior to `gh pr create|merge` or `scripts/github_pr_helper.py --create|--merge`.
- **Part 1 — Maker (Author)**:
  1. Implement focused changes with explicit type annotations, docstrings on public APIs, and regression/negative tests.
  2. Update `CHANGELOG.md` (`## [Unreleased]` or `## [X.Y.Z]`), `lessons_learned.md`, and `docs/plans/implementation_plan.md` when applicable.
  3. Pass `make local-ci` (and `make smoke-pr` during rapid iteration).
  4. Push the branch and open a PR using `.github/pull_request_template.md`, completing `## Summary`, `## Linked Issue`, and `## Maker Checklist` truthfully from observed output.
- **Part 2 — Checker (Independent Auditor)**:
  1. Inspect `git diff origin/main...HEAD` and `git status --short` for edge cases, exception safety, workspace/path containment (`x-workspace-id`, `mcp_server.py`), read-only Cypher enforcement (`GraphTools.graph_query`), `BaseGraphStore` parity (`KuzuStore` / `Neo4jStore`), and test rigor (no unspecced mocks masking broken signatures).
  2. Confirm `make local-ci` passes cleanly on the head commit and post findings in the PR `## Checker Review` comment/section.
  3. Squash-merge cleanly and delete the remote head branch via `scripts/github_pr_helper.py --merge --pr <N> --delete-branch` (or `gh pr merge <N> --squash --delete-branch`), then sync `main` (`git checkout main && git pull --ff-only origin main && git fetch --prune origin && git branch -d <branch>`).
- **Zero Lingering Open PR Invariant**: Before declaring work complete, execute `uv run --no-sync python scripts/github_pr_helper.py --audit` (`make audit-prs`) and verify it exits `0` (`OK: Zero open pull requests.` and `OK: Zero lingering remote feature branches.`).

---

## 3. Documentation Synchronicity & Release Tracking

- **Plans & Walkthroughs (`docs/plans/`)**: Keep the active hardening/implementation checklist in `docs/plans/implementation_plan.md` and record verified milestone walkthroughs in `docs/plans/latest_walkthrough.md`.
- **Lessons Learned (`lessons_learned.md`)**: Store debugging insights, test environment fixes, and architectural lessons exclusively in `lessons_learned.md`.
- **Continuous `CHANGELOG.md`**: Log every completed PR's scope, PR number, and observed coverage/test impact under the active release heading.

---

## 4. Test Timings Reporting, Hermetic Isolation & Cross-Platform Invocation

- **Timings & Metrics Summary**: At the end of verification or milestone execution, present a structured summary of local CI stage results, test counts, and coverage percentage.
- **Hermetic Isolation**: Every test must use temporary directories (`tmp_path` or `tempfile.TemporaryDirectory()`) or isolated in-memory stores — never leave stray database files (`*.db`, `*.db-wal`, `*.db-shm`) or store directories at the repository root.
- **Cross-Platform Invocation**: Prefer `uv run --no-sync python ...` (or the repository `.venv` Python) over bare system `python3`; use forward slashes (`/`) in Markdown links.

---

## 5. Clean Workspace Hygiene & Zero Local Path Leakage

- Keep the root clean of stray caches, `.db` files, and temporary test directories; preserve `.agents/skills/` and `.agents/skills.json`.
- Zero tracked `*.pyc` files (`git ls-files "*.pyc"` must be empty) and zero committed SQLite DB artifacts.
- **Zero Machine-Local Paths**: Never write local workstation filesystem paths or `file://` URLs into committed source, configuration, documentation, `.github/`, `.agents/`, `deploy/`, GitHub issues, PR descriptions, or PR comments.

---

## 6. SemVer Parity & Code-First Invariants

- **SemVer Parity**: `[project].version` in `pyproject.toml`, `__version__` (and fallback literals) in `knowledge_graph_pkg/__init__.py`, `CITATION.cff`, `.agents/skills/knowledge-reduce-core/SKILL.md`, and `scripts/smoke_skill.py` MUST match 1:1. Tag releases cleanly on `main` (`git tag -a vX.Y.Z -m "..."`).
- **Code-First Invariants**: Core invariants (workflow trigger checks, full MIT `LICENSE`, SemVer parity, path/pyc hygiene, read-only Cypher guards, `GraphTools` schema parity, offline `FactCritic` heuristics, `--cov-fail-under=75` parity) are enforced mechanically in `tests/meta/test_governance_and_invariants.py`.

---

## 7. Referenced Rule Documents

- `.agents/rules/coding_method_guardrails.md` — 10 mandatory coding-method guardrails for extraction, graph stores, security, and test rigor.
- `.agents/rules/release_and_pr_management.md` — Release tagging, SemVer parity, `CHANGELOG.md`, `docs/plans/`, and Maker/Checker PR lifecycle procedures.
