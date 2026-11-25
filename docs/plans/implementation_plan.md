# Implementation Plan: Codebase Audit, Hardening & Quality Assurance (`v0.4.0`)

Master roadmap archived in `docs/plans/master_plan_v0_4_0.md`.

## Phase 1: Security Hardening, Graph Store Parity, MCP Server & Watcher Pipeline (`PR #2`)
- `[x]` Graph store parity (`graph_store_base.py`, `kuzu_store.py`, `neo4j_store.py`, `graph_store_factory.py`)
- `[x]` `GraphTools` read-only Cypher guard & retriever caching (`graph_tool.py`)
- `[x]` `mcp_server.py` auth, workspace containment, WebSocket isolation, CORS, stdlib parity & `DASHBOARD_HTML` XSS/bug fixes
- `[x]` `WatcherDaemon` + `factory.batch_drop` real graph pipeline contract & atomic ontology output
- `[x]` Clean isolated unit & integration tests (`tests/test_neo4j_store.py`, `tests/test_graph_tool.py`, `tests/test_mcp_server.py`, `tests/test_watcher.py`, `tests/test_reasoning.py`)
- `[ ]` Maker/Checker verification (`make local-ci`, `make smoke-pr`, `make audit-prs`) & merge PR #2

## Remaining Phases (`v0.4.0`)
- `[ ]` Phase 2: Entity Resolution, Graph-RAG, Graph Compiler & Ontology Distillation (`PR #3`)
- `[ ]` Phase 3: ModelReduce Refinery, Probing, Consensus, Critique, Crawler & Training (`PR #4`)
- `[ ]` Phase 4: Core Graph, V1 Ingestion/Distillation, Storage & Legacy Modules (`PR #5`)
- `[ ]` Phase 5: CLI Packaging & Error Handling, Scripts, Examples & Test Isolation (`PR #6`)
- `[ ]` Phase 6: Coverage Hardening, Gold Eval & `v0.4.0` SemVer/Doc Sync (`PR #7`)
