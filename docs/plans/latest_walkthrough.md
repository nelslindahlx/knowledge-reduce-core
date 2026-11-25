# Latest Walkthrough: Phase 2 Entity Resolution, Graph-RAG, Compiler & Ontology Distillation

Summary of Phase 2 remediation from `docs/plans/master_plan_v0_4_0.md` executed on branch `fix/graph-rag-er-ontology`.

## Changes Applied
- **Graph Store Edge Counter (`graph_store_base.py`)**:
  - Added `BaseGraphStore.edge_count()` returning the count of `RELATED` edges so caching layers can detect relationship mutations even when node counts remain constant.
- **Boundary-Aware, Edge-Preserving Entity Resolution (`entity_resolution.py`)**:
  - Replaced arbitrary substring merging (`s1_clean in s2_clean`) with exact normalized matching, acronym/parenthetical matching (`"ATP"` <-> `"Adenosine Triphosphate (ATP)"`), initialism matching (`"LLM"` <-> `"Large Language Model"`, `"USA"` <-> `"United States of America"`), and long chemical-prefix normalization (`"Glucose"` <-> `"D-Glucose (Sugar)"`), preventing false merges such as `"AI"` -> `"Brain Activation"` or `"Fe"` -> `"Infection"`.
  - Replaced destructive full-graph `DETACH DELETE` with change-only updates: writes merged nodes first, remaps affected `RELATED` edges via `MERGE ... ON CREATE SET`, and deletes only obsolete `block_id`s while keeping untouched subgraphs and edges intact.
  - Ranked duplicate collapse candidates by `(RELIABILITY_RANK, agreement, quality)`, merged deduplicated `source_models`, stored `int` quality for `INT64` schema parity, rewrote statements on word boundaries (preserving `"ATPase"`), and shared `_block_id` / `_s`.
- **Contradiction-Aware Graph-RAG (`rag.py`)**:
  - Updated conflict pruning to ignore resolved `UNVERIFIED` rows and require `_core_predicates_match()` on real store records so valid multi-predicate facts are preserved.
  - Pushed `exclude_unverified` filtering into `retrieve_seeds()` before `limit`, added query/statement vector caching (`_embedding_cache`, cached `VectorIndex`, `invalidate_cache()`), invalidated PageRank via `(store.count(), store.edge_count())`, and added pure-Python `_compute_pagerank_power()` fallback when `scipy` is not installed.
  - Added token-boundary matching for short 2–3 char terms (`"AI"`, `"Fe"`), `None`-safe field handling, `format_facts()`, and `exclude_unverified` forwarding in `format_context()`.
- **Graph Instruction Compiler (`graph_compiler.py`)**:
  - Added `auto_link` control and `LIMIT` pushdown, excluded `A -> B -> A` self-cycles, defaulted missing reliability to `"UNVERIFIED"`, required `_core_predicates_match()` for contradiction instructions with case-insensitive `RELIABILITY_RANK`, and wrote JSONL outputs atomically.
- **Single-Pass Case-Insensitive Ontology (`ontology.py` & callers)**:
  - Normalized predicates case-insensitively, stripped `None` values via `_s`, checked predicate rules before `-ing`/`-ation`/`-sis` suffixes, and added `distill_all()` + optional pre-fetched `facts` wired into `graph_tool.py`, `watcher.py`, `mcp_server.py`, and `cli.py`.
- **Expanded Tests (`tests/test_entity_resolution.py`, `tests/test_rag.py`, `tests/test_graph_compiler.py`, `tests/test_ontology.py`)**:
  - Replaced module-level `importorskip("kuzu")` in `test_entity_resolution.py`, `test_graph_compiler.py`, and `test_ontology.py` with pure-Python test classes plus `@pytest.mark.skipif` Kùzu integration classes, and added Kùzu end-to-end Graph-RAG integration tests.

## Verification Results

| Check | Result | Timing |
|---|---|---|
| `make local-ci` (`canary`, `skill`, `lint`, `test`) | `370 passed, 7 skipped`, `70.65%` coverage (`>= 58%` gate) | `5.32s` (`pytest`) |
| `make smoke-pr` (`canary`, `skill`, `lint`, Stage 1 & 2) | Stage 1 & Stage 2 passed cleanly (`1` optional skip) | Passed |
| `make canary` (`-W error::Warning` import + `tests/meta/`) | `10 passed`, zero import warnings | `< 1s` |
| `make lint` (`ruff check` + strict governance check/format) | `All checks passed!`, `4 files already formatted` | `< 1s` |
