# Latest Walkthrough: Phase 4 Core Graph, V1 Pipeline, Storage & Extended Modules

Summary of Phase 4 remediation from `docs/plans/master_plan_v0_4_0.md` executed on branch `fix/core-v1-legacy-modules`.

## Changes Applied
- **Storage, Lifecycle, Catalog & Factory (`store.py`, `lifecycle.py`, `catalog.py`, `factory.py`, `factstore.py`)**:
  - Updated `KnowledgeStore.audit_summary()` to recognize both canonical (`reliability_rating`, `fact_statement`) and legacy (`reliability`, `statement`) keys; added path-containment and ID/date sanitization in `_shard_path`/`iter_facts`, atomic shard/manifest writes, `{**safe, "record": "fact"}` collision protection, non-overwriting shard versioning (`-v2`), and `source=None` safety in `stats()`.
  - Made `reextract_store()` skip derived `-reextract-` drops and missing shards, normalize reliability/subject/predicate matching in `promote_reliability()` and `find_contradictions()`, and avoid `None` sorting crashes.
  - Added context manager support, parameterized `LIMIT ?`, case-insensitive exact `reliability` plus `min_reliability` ladder queries, and `NULL`-safe `stats()` to `Catalog`; made `scan_folder()` raise `FileNotFoundError` consistently on missing dirs, fixed `format_report` falsy-zero counts, and added atomic normalized persistence to `FactStore`.
- **Extraction, Semantic, Core & Distillation (`spacy_extractor.py`, `semantic.py`, `core.py`, `extraction.py`, `coref.py`, `evaluation.py`, `distillation.py`, `quality.py`, `qa.py`, `ingest.py`, `export.py`)**:
  - Enabled spaCy's lemmatizer (`disable=["ner"]`) with safe fallback, stored `subject_type`/`object_type` on nodes, removed the broken `0.405` similarity cap in `SemanticKnowledgeGraph._calculate_semantic_similarity()`, prevented `fact_id` collisions on repeated `source_id` runs, and expanded predicate verbalizations.
  - Made `KnowledgeGraph.update_fact()` validate all keys and types atomically before mutating, normalize `tags`/`datetime` types consistently, and raise `ValueError` directly.
  - Hardened `SVOExtractor` copula folding against relative clauses while preserving `"Civic engagement..."` and `"Johnson & Johnson"`, filtered sentence-opener stopwords from `coref.py` while keeping `"The Beatles"`, prevented `""` and short `"is"` stems from false-matching predicates in `evaluation.py`, enforced reliability/null guards and atomic writes in `KnowledgeDistiller`/`FactQualityFilter`/`QAGenerator`, preserved `snake_case` underscores in `markdown_to_text`, separated `</td>`/`</th>` table cells in HTML ingestion, and returned `0` for empty `estimate_tokens("")`.
- **Extended & Legacy Modules (`enhanced.py`, `utils.py`, `blockchain.py`, `streaming.py`, `sharding.py`, `vector.py`, `analysis.py`, `visualization.py`)**:
  - Replaced method-level `@lru_cache` with toggleable per-instance caches in `EnhancedKnowledgeGraph`, prevented dict mutation in `batch_add_facts()`, and fixed `save_to_file()`/`load_from_file()` round-trips (`quality_score`, comma-string `tags`).
  - Made `utils.py` import `requests`/`bs4` lazily, validate `http`/`https` schemes and request timeouts, and round-trip JSON/CSV/filter/merge safely via `_add_node_data_to_kg()`.
  - Fixed `BlockchainKnowledgeGraph` JSON serialization, caller dict immutability, normalized tag comparison in `verify_fact()`, PoW/genesis validation in `verify_chain()`, and chain preservation on invalid `import_chain()`.
  - Added `RLock` synchronization, `finally: task_done()`, full field defaults, and `limit <= 0` handling in `StreamingKnowledgeGraph`; added `shard_size` validation, SHA-256 hashing, deterministic ordering, and stale-file cleanup in `ShardedKnowledgeGraph.optimize_shards()`; aligned `VectorKnowledgeGraph` vector spaces, deterministic SHA-256 embeddings, and small-corpus clustering; fixed tag whitespace and mixed reliability handling in `analysis.py`; and made `visualization.py` import `matplotlib` lazily while sanitizing GEXF/GraphML attributes, guarding empty graphs, and mapping reliability colors by tier name.
- **Expanded Tests**:
  - Expanded 18 Cluster A/analysis/visualization test modules and added 6 new Cluster B test suites (`tests/test_enhanced.py`, `tests/test_utils.py`, `tests/test_blockchain.py`, `tests/test_streaming.py`, `tests/test_sharding.py`, `tests/test_vector_kg.py`) with zero CWD leftovers or `sys.modules` leaks.

## Verification Results

| Check | Result | Timing |
|---|---|---|
| `knowledgereduce eval --gold data/gold_set.json` | `P=0.800, R=0.923, F1=0.857` (`12 TP, 3 FP, 1 FN`) | `< 1s` |
| `make local-ci` (`canary`, `skill`, `lint`, `test`) | `433 passed, 5 skipped`, `84.53%` coverage (`>= 58%` gate) | `6.06s` (`pytest`) |
| `make smoke-pr` (`canary`, `skill`, `lint`, Stage 1 & 2) | All stages passed cleanly | Passed |
| `make lint` (`ruff check` + strict governance check/format) | `All checks passed!`, `4 files already formatted` | `< 1s` |
