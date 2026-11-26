# Implementation Plan: Codebase Audit, Hardening & Quality Assurance (`v0.4.0`)

Master roadmap archived in `docs/plans/master_plan_v0_4_0.md`.

## Phase 1: Security Hardening, Graph Store Parity, MCP Server & Watcher Pipeline (`PR #2`)
- `[x]` Graph store parity (`graph_store_base.py`, `kuzu_store.py`, `neo4j_store.py`, `graph_store_factory.py`)
- `[x]` `GraphTools` read-only Cypher guard & retriever caching (`graph_tool.py`)
- `[x]` `mcp_server.py` auth, workspace containment, WebSocket isolation, CORS, stdlib parity & `DASHBOARD_HTML` XSS/bug fixes
- `[x]` `WatcherDaemon` + `factory.batch_drop` real graph pipeline contract & atomic ontology output
- `[x]` Clean isolated unit & integration tests (`tests/test_neo4j_store.py`, `tests/test_graph_tool.py`, `tests/test_mcp_server.py`, `tests/test_watcher.py`, `tests/test_reasoning.py`)
- `[x]` Maker/Checker verification (`make local-ci`, `make smoke-pr`, `make audit-prs`) & merged PR #2 (`455e322`)

## Phase 2: Entity Resolution, Graph-RAG, Graph Compiler & Ontology Distillation (`PR #3`)
- `[x]` `BaseGraphStore.edge_count()` (`graph_store_base.py`)
- `[x]` Non-destructive, boundary-aware entity resolution (`entity_resolution.py`)
- `[x]` Contradiction-aware `GraphRAGRetriever`, vector/PageRank caching, power-iteration fallback & `format_facts()` (`rag.py`)
- `[x]` Cycle-free subgraph & true-contradiction `graph_compiler.py`
- `[x]` Case-insensitive single-pass `OntologyDistiller` (`ontology.py`) wired into `graph_tool.py`, `watcher.py`, `mcp_server.py`, `cli.py`
- `[x]` Pure-Python + `KuzuStore` tests (`tests/test_entity_resolution.py`, `tests/test_rag.py`, `tests/test_graph_compiler.py`, `tests/test_ontology.py`)
- `[x]` Maker/Checker verification (`make local-ci`, `make smoke-pr`, `make audit-prs`) & merged PR #3 (`9eae4e2`)

## Phase 3: ModelReduce Refinery, Probing, Consensus, Critique, Crawler & Training (`PR #4`)
- `[x]` Structured probing, schemas, Gemini sanitization, timeouts & logprob guards (`model_probe.py`, `probe_templates.py`)
- `[x]` `FactCritic` aligned with real backend protocol, fail-closed errors & strict bool parsing (`critique.py`)
- `[x]` Multi-engine `ConsensusEngine` + `factory.batch_drop(include_engine_in_hash=True)` & Neo4j URI parity (`consensus.py`, `factory.py`)
- `[x]` Boundary-matched capability weights, empty-SPO guard, distillation parity & compound-safe `ModelDrop` (`cross_model.py`, `model_distill.py`, `model_drop.py`)
- `[x]` Exact gold recall, negation-aware eval, crawler guards, atomic graveyard & MLX/SFT prep (`embeddings.py`, `model_eval.py`, `crawler.py`, `graveyard.py`, `training_prep.py`, `train.py`)
- `[x]` Upgraded ModelReduce test suites & eliminated `test_train_wrapper.py` / `test_model_probe.py` `sys.modules` leaks
- `[x]` Maker/Checker verification (`make local-ci`, `make smoke-pr`, `make audit-prs`) & merged PR #4 (`8d708ba`)

## Phase 4: Core Graph, V1 Ingestion/Distillation, Storage & Extended Modules (`PR #5`)
- `[x]` Hardened V1 store, catalog, lifecycle, factory & factstore (`store.py`, `catalog.py`, `lifecycle.py`, `factory.py`, `factstore.py`)
- `[x]` Fixed extraction, lemmatization, semantic similarity, core updates, coref, evaluation & distillation (`spacy_extractor.py`, `semantic.py`, `core.py`, `extraction.py`, `coref.py`, `evaluation.py`, `distillation.py`, `quality.py`, `qa.py`, `ingest.py`, `export.py`)
- `[x]` Hardened all 8 extended modules (`enhanced.py`, `utils.py`, `blockchain.py`, `streaming.py`, `sharding.py`, `vector.py`, `analysis.py`, `visualization.py`)
- `[x]` Expanded Cluster A tests and added 6 Cluster B test suites (`84.53%` total coverage, gold F1 `0.857`)
- `[x]` Maker/Checker verification (`make local-ci`, `make smoke-pr`, `make audit-prs`) & merged PR #5 (`2bbadc4`)

## Phase 5: CLI Packaging & Error Handling, Scripts, Examples & Test Isolation (`PR #6`)
- `[x]` Hardened `knowledge_graph_pkg/cli.py` (packaged `_import_script`, dead `ImportError` fixes, `.close()` guards, `catalog --min-quality 0`, single-pass `query-graph` & `distill`, atomic outputs, exit codes)
- `[x]` Hardened all scripts (`run_suite.py`, `download_and_crawl.py`, `test_drive.py`, `mirror_skill.py`, `train_mlx.py`, `train_sft.py`)
- `[x]` Fixed and verified all 7 `examples/*.py` + `scripts/test_drive.py` offline with zero CWD leftovers
- `[x]` Isolated `tests/test_compile_sft.py`, lazy `find_spec` in `tests/conftest.py`, expanded `tests/test_cli.py` (`443 passed, 5 skipped`, `86.23%` coverage)
- `[x]` Maker/Checker verification (`make local-ci`, `make smoke-pr`, `make audit-prs`) & merged PR #6 (`af44af4`)

## Phase 6: Coverage Hardening, Gold Eval & `v0.4.0` SemVer/Doc Sync (`PR #7`)
- `[x]` Ratcheted coverage gate from `--cov-fail-under=58` to `--cov-fail-under=75` across all 7 governance parity files (`Makefile`, `AGENTS.md`, `.agents/AGENTS.md`, `CONTRIBUTING.md`, `.github/pull_request_template.md`, `.github/workflows/ci.yml`, `tests/meta/test_governance_and_invariants.py`) and `README.md`
- `[x]` Bumped SemVer cleanly from `0.3.0` to `0.4.0` across `pyproject.toml`, `knowledge_graph_pkg/__init__.py`, `CITATION.cff`, `.agents/skills/knowledge-reduce-core/SKILL.md`, and `scripts/smoke_skill.py` (with editable `.venv` metadata refreshed offline)
- `[x]` Synchronized `ARCHITECTURE.md`, `README.md`, `CHANGELOG.md`, `lessons_learned.md`, and `docs/plans/latest_walkthrough.md` with zero machine-local path leakage
- `[x]` Verified gold evaluation (`knowledgereduce eval --gold data/gold_set.json`, `ModelShardEvaluator` + `data/gold_biochem.json`), all 7 `examples/*.py`, `scripts/test_drive.py`, simulated minimal-env coverage (`>= 75%`), `make local-ci`, `make smoke-pr`, and `make audit-prs` (`PR #7`)
