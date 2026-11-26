# Contributing to KnowledgeReduce Core

**KnowledgeReduce Core** enforces a **Maker / Checker Pull Request** workflow on `main` paired with **zero-GitHub-Actions-minutes stratified local CI**. Direct pushes to `main` are blocked; all changes flow through ephemeral feature branches, complete local CI verification, independent Checker diff audits, and clean PR merges.

---

## 1. Branch & Workspace Initialization

### Prerequisites
* **Python 3.9+** (3.11+ recommended)
* `uv` or standard `venv`

### Setup
```bash
git checkout main
git pull --ff-only origin main
git checkout -b <type>/<short-description>   # e.g. feat/graph-fix, fix/watcher-lock, chore/hardening
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e ".[dev,hermes]"
```

Optional extras for full NLP, ingestion, visualization, and graph databases:
```bash
.venv/bin/pip install -e ".[ingest,pdf,viz,model-reduce,graph]"
.venv/bin/python -m spacy download en_core_web_sm
python3 scripts/mirror_skill.py
```

Always confirm `git branch --show-current` is **not** `main` before committing, and never reuse a previously merged branch name.

---

## 2. Stratified Local CI (Zero GitHub Action Minutes)

Automatic GitHub Actions triggers (`push`, `pull_request`) are disabled (`workflow_dispatch` only) so routine development consumes **zero** GitHub Actions minutes. Run verification locally using the `Makefile` targets:

```bash
make canary      # <10s: import warning gate + tests/meta/ invariants & governance checks
make skill       # Hermes skill smoke test + tests/test_skill.py manifest validation
make lint        # ruff check across code/tests/scripts + strict ruff check/format on governance/meta modules
make smoke-pr    # canary + skill + lint + Stage 1 & Stage 2 tests
make test        # full pytest suite with --cov-fail-under=75
make local-ci    # complete local CI gate (canary + skill + lint + test)
make audit-prs   # verify zero open PRs and zero lingering remote branches
```

Cross-platform invocations also work directly via `uv run --no-sync python`:
```bash
uv run --no-sync python -W error::Warning -c "import knowledge_graph_pkg"
uv run --no-sync python -m pytest tests/meta/ -q
uv run --no-sync python scripts/smoke_skill.py
uv run --no-sync python -m ruff check knowledge_graph_pkg tests scripts
uv run --no-sync python -m pytest --cov=knowledge_graph_pkg --cov-report=term-missing --cov-fail-under=75
```

You can also run targeted layers with `scripts/run_suite.py`:
- **Stage 1**: Fast Unit, Core Graph, Schemas, Meta & Skill Gates (`scripts/run_suite.py --stage 1`)
- **Stage 2**: Semantic & Ingestion / Extraction (`scripts/run_suite.py --stage 2`)
- **Stage 3**: Lifecycle, Database & Graph Integration (`scripts/run_suite.py --stage 3`)
- **Stage 4**: ModelReduce Probes, Cross-Model & Evaluation (`scripts/run_suite.py --stage 4`)
- **All Stages**: `scripts/run_suite.py --stage all`

---

## 3. Maker / Checker PR Lifecycle

### Stage A — Maker (Authoring & Local CI)
1. Implement focused changes with explicit type annotations, docstrings on new public APIs, and regression/negative tests.
2. Update `CHANGELOG.md`, `lessons_learned.md`, and `docs/plans/implementation_plan.md` when adding features, fixing subtle bugs, or completing hardening milestones.
3. Run `make local-ci` (and `make smoke-pr` during rapid iteration).
4. Push the feature branch and open a PR using `.github/pull_request_template.md`, completing `## Summary`, `## Linked Issue`, and `## Maker Checklist` truthfully from observed command outputs (with zero machine-local paths).

### Stage B — Checker (Audit, Merge & Zero-Lingering Check)
1. Audit `git diff origin/main...HEAD` for correctness, workspace/path containment (`x-workspace-id` in `mcp_server.py`), JWT validation, read-only Cypher guards (`GraphTools.graph_query`), `BaseGraphStore` / backend contract parity, and test rigor.
2. Confirm `make local-ci` passed on the head commit and record the findings under `## Checker Review` on the PR.
3. Merge and clean up the branch via:
   ```bash
   uv run --no-sync python scripts/github_pr_helper.py --merge --pr <PR_NUMBER> --delete-branch
   git checkout main && git pull --ff-only origin main
   git fetch --prune origin
   uv run --no-sync python scripts/github_pr_helper.py --audit
   ```

---

## 4. Pre-Commit Hooks & Style

Install pre-commit hooks locally:
```bash
.venv/bin/pip install pre-commit
.venv/bin/pre-commit install
```

Hooks enforce trailing whitespace cleanup, end-of-file newlines, YAML checks, private-key detection, Ruff lint fixes, and Ruff formatting. Keep all unrelated docstrings and comments intact, preserve `.agents/skills/`, and never commit machine-local paths or `*.db` / `*.pyc` artifacts.

---

## 5. Releases

Version strings in `pyproject.toml`, `knowledge_graph_pkg/__init__.py`, `CITATION.cff`, `.agents/skills/knowledge-reduce-core/SKILL.md`, and `scripts/smoke_skill.py` must match 1:1. Annotated release tags (`vX.Y.Z`) are created from merged commits on `main` after `make local-ci` and `--audit` pass cleanly. See `.github/workflows/release.yml` for manual `workflow_dispatch` PyPI Trusted Publisher releases.
