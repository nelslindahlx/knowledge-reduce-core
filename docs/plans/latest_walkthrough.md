# Latest Walkthrough: Repository Hardening

Summary of repository hardening ported from `KnowledgeReduce` into `knowledge-reduce-core` on branch `chore/repo-hardening`.

## Hardening Scope
- **Packaging & Metadata**: Completed MIT `LICENSE`, added `CITATION.cff` (`0.3.0`), synchronized `knowledge_graph_pkg/__init__.py` fallback `__version__` (`"0.3.0"`), added Ruff & coverage settings (`--cov-fail-under=58` parity) in `pyproject.toml`, modernized `.pre-commit-config.yaml`, and fixed 6 test lint errors (`F821`, `F811`, `F841`).
- **Path & Artifact Hygiene**: Scrubbed all machine-local paths across deployment manifests and planning docs, fixed `<string>` argument entries in `deploy/com.knowledgereduce.watcher.plist`, and expanded `.gitignore`.
- **Agent Governance & Hooks**: Added `AGENTS.md` + `.agents/AGENTS.md` (byte-identical), `.agents/rules/`, `.agents/hooks.json`, `scripts/hooks/pre_pr_safety_gate.py`, and `scripts/github_pr_helper.py`.
- **Zero-Minutes Workflows & Local CI**: Configured `.github/workflows/*.yml` (`ci.yml`, `pr-governance.yml`, `release.yml`, `skill-validation.yml`) for `workflow_dispatch` only, added `.github/CODEOWNERS` and `.github/pull_request_template.md`, created `Makefile`, and added mechanical meta-tests in `tests/meta/test_governance_and_invariants.py`.

## Verification Results

| Check | Result | Timing |
|---|---|---|
| `make local-ci` (`canary`, `skill`, `lint`, `test`) | `308 passed, 16 skipped`, `59.62%` coverage (`>= 58%` gate) | `2.82s` (`pytest`: `1.66s`) |
| `make smoke-pr` (`canary`, `skill`, `lint`, Stage 1 & 2) | Stage 1: `68 passed`; Stage 2: `68 passed, 1 skipped` | `1.91s` |
| `make canary` (`-W error::Warning` import + `tests/meta/`) | `10 passed`, zero import warnings | `< 1s` |
| `make skill` (`smoke_skill.py` + `tests/test_skill.py`) | `SUCCESS`, `2 passed` | `< 1s` |
| `make lint` (`ruff check` + strict governance check/format) | `All checks passed!`, `4 files already formatted` | `< 1s` |
| PreToolUse Hook (`scripts/hooks/pre_pr_safety_gate.py`) | Denied `git push origin main`; allowed feature branch push | `< 1s` |
| `git ls-files "*.pyc"` & local-path grep | Empty / zero matches | `< 1s` |
