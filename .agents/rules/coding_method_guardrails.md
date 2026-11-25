# Mandatory Coding-Method Guardrails

Every contributor and AI agent authoring, modifying, or reviewing code in **KnowledgeReduce Core** should apply this 10-point engineering checklist before opening a Pull Request.

---

## 1. End-to-End Producer / Consumer Contract Verification
- Every change across the pipeline (`ingest -> extraction/spacy -> KnowledgeStore -> BaseGraphStore -> critique/cross_model -> GraphRAGRetriever -> export/SFT`) **must** verify that downstream consumers read the exact dict keys, schemas, and method signatures emitted upstream.

## 2. Backend & Graph Store Interface Parity
- Keep `BaseGraphStore` (`KuzuStore`, `Neo4jStore`) and model probing backends (`OllamaBackend`, `LlamaCppBackend`, `OpenAICompatibleBackend`, `GeminiBackend`) aligned to shared contracts (`ingest_facts`, `query`, `generate_structured`, `generate_text_with_logprobs`) without mismatched field names or unimplemented method calls.

## 3. Provenance & Reliability-Rating Preservation
- Preserve `ReliabilityRating` tiers (`UNVERIFIED`, `POSSIBLY_TRUE`, `LIKELY_TRUE`, `VERIFIED`), `source_models`, and `block_id` identity across `Drop` shards, consensus merging, and graph ingestion; keep `exclude_unverified=True` filtering intact in `GraphRAGRetriever` and `compile_subgraph_instructions`.

## 4. Explicit Error Visibility Over Silent Swallowing
- Avoid bare `except Exception` fallbacks that silently mask broken backend calls or invalid schemas as positive results; surface actionable diagnostics or structured `{"ok": False, "error": ...}` payloads cleanly.

## 5. Bounded Memory, Caching & Blocking Caps
- Keep PageRank caches (`GraphRAGRetriever.calculate_pagerank`), embedding calculations, SQLite WAL state (`WatcherDaemon`), and prefix-blocked `entity_resolution.py` merges bounded and cache-invalidated whenever graph counts mutate.

## 6. Security, Auth & Path Containment
- Sanitize HTTP headers (`x-workspace-id` in `mcp_server.py`) strictly against path traversal and URI injection before opening graph stores.
- Require valid PyJWT signature + `exp` verification on protected HTTP/WebSocket endpoints when `MCP_JWT_SECRET` is configured, never accept raw secrets as bypass tokens, and enforce parameterized Cypher queries plus read-only guards in `GraphTools.graph_query`.

## 7. Optional-Dependency Isolation & Clean Teardown
- Keep optional packages (`kuzu`, `neo4j`, `spacy`, `matplotlib`, `fastapi`, `watchdog`, `mlx`, `llama_cpp`, `ollama`, `google.generativeai`) lazy-imported behind extras and gated by `tests/conftest.py` markers or `pytest.importorskip(...)`.
- Ensure tests use `tmp_path` or `tempfile.TemporaryDirectory()` and close `KuzuStore` / `sqlite3` handles cleanly before teardown.

## 8. Pre-Edit Meta Gate & Skill Smoke Verification
- Before modifying extraction, graph reasoning, MCP tool schemas, or packaging metadata, run `make canary` and `make skill` to verify governance invariants and Hermes skill compatibility remain intact.

## 9. Shared Helpers & Specced Test Mocks
- Consolidate duplicate logic (`_block_id`, `get_graph_store`, `get_backend`, `handle_tool_call`) rather than duplicating behavior across CLI, MCP, and watcher daemons.
- When mocking backends or stores in unit tests, use `create_autospec` / `spec=` against real classes where possible so signature drift is caught immediately.

## 10. Explicit `is not None` Numeric & Threshold Checks
- Always use `if value is not None:` for optional numeric thresholds (`similarity_threshold`, `top_k`, `hops`, `logprob_threshold`, `quality_score`) so valid `0` or `0.0` values are never overwritten by truthiness fallbacks.
