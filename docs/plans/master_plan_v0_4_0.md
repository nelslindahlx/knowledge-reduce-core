# Master Plan v0.4.0: Full-Codebase Audit, Hardening & Quality Assurance

This plan tracks the 6-phase remediation and verification program derived from a complete line-by-line audit of `knowledge_graph_pkg/`, `scripts/`, `examples/`, and `tests/`. Each phase ships via an isolated feature branch and Maker/Checker Pull Request targeting `main`, verified locally with `make local-ci`.

---

## Phase 1 — Security Hardening, Graph Store Parity, MCP Server/Dashboard & Watcher Pipeline (`PR #2`)
- **Graph Store Parity (`graph_store_base.py`, `kuzu_store.py`, `neo4j_store.py`, `graph_store_factory.py`)**:
  - Extract shared `_s`, `_block_id`, `RELIABILITY_RANK`, and `normalize_fact_record` accepting both canonical (`fact_statement`, `reliability_rating`, `cross_model_agreement`, `quality_score`, list `source_models`) and legacy keys.
  - Enforce `MERGE ... SET` parity in `Neo4jStore`, remove hardcoded `"password"` default in favor of `NEO4J_PASSWORD` / explicit arg, fix credential leakage in `clean_uri`, add `bolt+s://` / `bolt+ssc://`, and add `Neo4jStore.with_database()`.
  - Fix `BaseGraphStore` Cypher & reasoning logic (`find_by_reliability` clamp, `auto_link_relations` empty-field guard + non-clobbering `ON CREATE SET`, `find_contradictions` core-predicate overlap + null guards, `find_transitive_inferences` case-insensitivity + `UNVERIFIED`/self-loop exclusions, `validate_and_reconcile` tier+quality ranking + deduplicated demotions).
- **Cypher Guard (`graph_tool.py`)**:
  - Strip quoted string literals before checking forbidden keywords (`CALL`, `FOREACH`, `ATTACH`, `EXPORT`, `IMPORT`, `USE`, `GRANT`/`DENY`/`REVOKE`, etc.), embedded semicolons, and comments; wire `self.default_limit`, enforce Python slice cap, and reuse cached `GraphRAGRetriever`.
- **MCP Server & Dashboard (`mcp_server.py`)**:
  - Enforce `x-workspace-id` regex allowlist (`[A-Za-z0-9_-]{1,64}`), path containment (`os.path.commonpath`) using `KuzuStore.path`, LRU cache eviction with `.close()`, and `Neo4jStore.with_database()`.
  - Remove raw `token == secret` fallback in `verify_token`, require JWT auth on `POST /api/notify_update` and `WS /ws`, partition WebSockets by workspace, restrict CORS defaults, validate JSON dict payloads, add auth + body-size checks in stdlib `make_handler`, fix `/api/graph` `.strip()`, and eliminate duplicate `retrieve()` in `/api/rag_retrieve`.
  - Fix embedded `DASHBOARD_HTML` XSS (`escapeHtml` + `data-block-id` event listeners instead of inline `onclick`), undefined `token` / `edgesDataset` `ReferenceError`s, WebGL renderer reuse, RAG highlight callback preservation, camera zero-division guard, and null/array safety.
- **Watcher Pipeline (`factory.py`, `watcher.py`)**:
  - Emit `"fact_records": drop.facts` in `batch_drop`, normalize `reliability` case-insensitively, preserve Neo4j URIs in `WatcherDaemon.__init__`, add `on_moved` and lock synchronization, compute `mtime` after write-settle wait, run graph ingestion/resolution/linking/validation before logging `PROCESSED`, and write `ontology_summary.json` atomically.
- **Test Hygiene**:
  - Eliminate repo-root CWD artifact creation and module-level `sys.modules` overwrites across `tests/test_watcher.py`, `tests/test_neo4j_store.py`, `tests/test_mcp_server.py`, `tests/test_graph_tool.py`, and `tests/test_reasoning.py`, plus add real integration tests without mocking `batch_drop`.

---

## Phase 2 — Entity Resolution, Graph-RAG, Graph Compiler & Ontology Distillation (`PR #3`)
- Fix destructive delete-all-and-reinsert in `entity_resolution.py` (preserve `RELATED` edges, update only affected facts, eliminate short-substring over-merging, sort duplicates by `(RELIABILITY_RANK, agreement, quality)`, and enforce `INT64` quality casts).
- Fix `GraphRAGRetriever` (`rag.py`): prune only true negating contradictions (excluding resolved `UNVERIFIED`), stop dropping `<= 2`-char technical terms, include `subject`/`object` in re-ranking, avoid duplicate `retrieve()` in `format_context` (add `format_facts`), invalidate PageRank cache on graph mutations, and cache statement vectors instead of re-embedding all facts per query.
- Fix `graph_compiler.py` (`compile_contradiction_instructions` using `store.find_contradictions()`, `compile_subgraph_instructions` defaulting missing reliability to `UNVERIFIED` and preventing `A -> B -> A` self-cycles).
- Fix `OntologyDistiller` (`ontology.py`): case-insensitive predicate matching, `None`-value guards, and single-pass query reuse.

---

## Phase 3 — ModelReduce Refinery, Probing, Consensus, Critique, Crawler & Training (`PR #4`)
- Fix `FactCritic` (`critique.py`) to call `generate_structured` / `generate_text_with_logprobs`, normalize JSON schemas, fail closed/explicit on backend errors, and guard `subj == obj` and `"false"` string booleans.
- Fix `model_probe.py`: OpenAI `json_schema` `response_format`, Gemini `google-genai` compatibility / strip unsupported schema bounds, `LlamaCppBackend` `logits_all` + `None` token logprob filtering, `get_backend` explicit `None` forwarding, and network timeouts across all clients.
- Fix `ConsensusEngine` (`consensus.py`), `cross_model.py` (inverted `ml in name` capability check + Ollama `:` delimiter + consistent `reliability_for_agreement`), `model_drop.py` domain path sanitization, `model_eval.py` recall math + predicate awareness, `embeddings.py` empty-vector/dimension guards + bounded cache, `crawler.py`, `graveyard.py` atomic checkpoints, `training_prep.py` in-place truncation/validation, and `train.py` `mlx_lm.lora` args.

---

## Phase 4 — Core Graph, V1 Ingestion/Distillation, Storage & Legacy Modules (`PR #5`)
- Fix `store.py` (`audit_summary` reading canonical `"fact_statement"`/`"reliability_rating"`, atomic `write_drop` + manifest updates, shard path containment), `lifecycle.py`, `catalog.py` (context manager, `>=` reliability filtering, `None` key safety), `factory.py`, and `factstore.py`.
- Fix `spacy_extractor.py` (re-enable lemmatizer or preserve verb lemmas), `semantic.py` (attach `subject_type`, fix IDF similarity cap, regex date bounds, unique `fact_id`), `extraction.py` (active+copula subject folding), `coref.py` (stop capturing pronouns/determiners), `evaluation.py` (empty/short-substring predicate matches), `distillation.py`, `quality.py`, `qa.py`, `ingest.py`, `export.py`, and `core.py`.
- Repair broken/untested legacy modules (`enhanced.py`, `utils.py`, `blockchain.py`, `streaming.py`, `sharding.py`, `vector.py`, `analysis.py`, `visualization.py`).

---

## Phase 5 — CLI Packaging, Scripts, Examples & Test Isolation (`PR #6`)
- Move CLI-called helpers inside `knowledge_graph_pkg/` so `train-mlx`, `train-sft`, `test-drive`, `run-suite`, and `skill-install` never fail with `ModuleNotFoundError: No module named 'scripts'` in packaged wheels.
- Fix dead `except ImportError` guards, missing DB `.close()` calls, exit codes, and `catalog`/`crawl`/`distill` edge cases in `cli.py`.
- Fix `scripts/*.py` (`download_and_crawl.py`, `train_mlx.py`, `train_sft.py`, `run_suite.py`, `test_drive.py`, `mirror_skill.py`, `smoke_skill.py`) and ensure all 7 `examples/*.py` run cleanly end-to-end without leaving CWD artifacts.
- Remove unused `conftest.py` markers / CWD writes (`test_compile_sft.py`) / stale `run_suite.py` stage mappings.

---

## Phase 6 — Coverage Hardening, Gold Eval & v0.4.0 Release Sync (`PR #7`)
- Raise test coverage across `knowledge_graph_pkg` (target >= 85% with optional extras and ratchet minimal-env `--cov-fail-under` parity).
- Verify `knowledgereduce eval` against `data/gold_set.json` and `data/gold_biochem.json`.
- Synchronize SemVer `0.4.0` across `pyproject.toml`, `knowledge_graph_pkg/__init__.py`, `CITATION.cff`, `.agents/skills/knowledge-reduce-core/SKILL.md`, `scripts/smoke_skill.py`, `ARCHITECTURE.md`, `README.md`, `CHANGELOG.md`, and `lessons_learned.md`.
