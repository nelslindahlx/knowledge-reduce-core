# Latest Walkthrough: Phase 1 Security, Store Parity, MCP & Watcher Hardening

Summary of Phase 1 remediation from `docs/plans/master_plan_v0_4_0.md` executed on branch `fix/security-mcp-stores-watcher`.

## Changes Applied
- **Graph Store Parity & Reasoning (`graph_store_base.py`, `kuzu_store.py`, `neo4j_store.py`, `graph_store_factory.py`)**:
  - Added shared `_s`, `_block_id`, `RELIABILITY_RANK`, and `normalize_fact_record` supporting canonical (`fact_statement`, `reliability_rating`, `cross_model_agreement`, `quality_score`, list `source_models`) and legacy schemas.
  - Switched `Neo4jStore` to idempotent `MERGE ... SET`, required explicit password / `NEO4J_PASSWORD`, added `Neo4jStore.with_database()`, added `bolt+s://` / `bolt+ssc://`, and stripped credentials from `clean_uri`.
  - Fixed `BaseGraphStore` Cypher & reasoning logic (`find_by_reliability` clamp, `auto_link_relations` empty guards + `ON CREATE SET`, `find_contradictions` core-predicate overlap, `find_transitive_inferences` case-insensitivity + `UNVERIFIED`/self-loop filtering, `validate_and_reconcile` tier+quality ranking).
- **Cypher Read-Only Guard (`graph_tool.py`)**:
  - Stripped quoted literals before scanning expanded `_FORBIDDEN` keywords (`CALL`, `FOREACH`, `EXPORT`, `IMPORT`, `ATTACH`, etc.), comments (`//`, `/*`), and multi-statement semicolons; wired `default_limit`, Python slice capping, and cached `GraphRAGRetriever`.
- **MCP Gateway & Dashboard (`mcp_server.py`)**:
  - Validated `x-workspace-id` via allowlist regex and `os.path.commonpath` against `KuzuStore.path`, added LRU workspace eviction with `.close()`, removed raw secret bearer bypass, enforced JWT auth on `/api/notify_update`, `WS /ws`, and stdlib `make_handler`, partitioned WebSockets by workspace, restricted CORS defaults, eliminated double `retrieve()` in `/api/rag_retrieve`, and fixed `DASHBOARD_HTML` XSS (`escapeHtml` + `data-block-id`), `edgesDataset`/`token` `ReferenceError`s, and WebGL renderer reuse.
- **Watcher Pipeline (`factory.py`, `watcher.py`)**:
  - Added `"fact_records"` to `batch_drop`, normalized `reliability` case-insensitively, preserved Neo4j URIs in `WatcherDaemon`, added `on_moved` + lock synchronization, sampled `mtime` after write-settle wait, executed graph ingestion/resolution/linking/validation before marking `PROCESSED`, and wrote `ontology_summary.json` atomically.
- **Test Isolation & Real Integration Coverage**:
  - Eliminated repo-root CWD database/file creation and module-level `sys.modules` pollution across `tests/test_watcher.py`, `tests/test_neo4j_store.py`, `tests/test_mcp_server.py`, `tests/test_graph_tool.py`, and `tests/test_reasoning.py`, and added a real end-to-end `WatcherDaemon` -> `KuzuStore` -> `OntologyDistiller` integration test.

## Verification Results

| Check | Result | Timing |
|---|---|---|
| `make local-ci` (`canary`, `skill`, `lint`, `test`) | `351 passed, 7 skipped`, `69.80%` coverage (`>= 58%` gate) | `11.00s` (`pytest`: `7.58s`) |
| `make smoke-pr` (`canary`, `skill`, `lint`, Stage 1 & 2) | Stage 1: `68 passed`; Stage 2: `71 passed, 1 skipped` | `5.14s` |
| `make canary` (`-W error::Warning` import + `tests/meta/`) | `10 passed`, zero import warnings | `< 1s` |
| `make lint` (`ruff check` + strict governance check/format) | `All checks passed!`, `4 files already formatted` | `< 1s` |
| `git ls-files "*.pyc"` & local-path grep | Empty / zero matches | `< 1s` |
