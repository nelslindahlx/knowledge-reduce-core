# Lessons Learned: KnowledgeReduce Project

This file contains a running registry of engineering insights, environment setup resolutions, and architectural discoveries made during the development of **KnowledgeReduce**.

---

## 1. Sandbox Caching & Execution Isolation

### Context
When setting up the local workspace inside a terminal sandbox, package installation and configuration files (such as `.gitconfig` or global system folders) were blocked by default sandboxing policies.
### Lesson Learned
* Direct network requests (like `git clone` or fetching packages from PyPI) require enabling sandbox bypass (`BypassSandbox: true`).
* Simple operations like virtualenv creation, file moves, and local test runs should be kept inside the sandbox (`BypassSandbox: false`) to ensure security and prevent accidental modifications to the user's home directories.

---

## 2. Legacy Pip limitations with Modern PEP 517 Packaging

### Context
A fresh Python 3.9 virtual environment initializes with `pip` version 21.2.4. Attempting to install the package in editable mode via `.venv/bin/pip install -e .` failed with `ERROR: File "setup.py" or "setup.cfg" not found.`.
### Lesson Learned
* Legacy pip versions require a `setup.py` or `setup.cfg` to install packages in editable mode, even if `pyproject.toml` is present with correct build-system declarations.
* Running `python3 -m pip install --upgrade pip` updates the environment to a modern pip version (e.g. 26.0+), which supports PEP 517 editable installations directly from `pyproject.toml` without any legacy files.

---

## 3. SpaCy Dependency Compilations & Version Matching

### Context
Attempting to install the `nlp` extra (which installs `spacy>=3.0.0`) failed because pip attempted to build `spacy` from source. The build failed due to compatibility restrictions in its sub-dependency `thinc` on Python 3.9.
### Lesson Learned
* Installing `spacy` without version constraints can result in pip trying to compile newer releases that are incompatible with the host Python version.
* Forcing a specific, widely-supported version like `spacy==3.7.5` matches available pre-compiled binary wheels for macOS arm64 on PyPI, completely bypassing compilation and installing successfully in under 5 seconds.

---

## 4. SpaCy Model Availability in Test Environments

### Context
After successfully installing the `spacy` package, running the test suite resulted in an `ImportError: spaCy model 'en_core_web_sm' not found.`.
### Lesson Learned
* Having the spaCy package installed does not automatically package or download default language models.
* The test environment must explicitly execute `python -m spacy download en_core_web_sm` to install the language model as a package before running NLP-dependent extractors.

---

## 5. Stage-Based Testing Benefits

### Context
Running the full test suite in one go executes all 30 test files, including database queries (Kùzu store) and skipped ModelReduce Ollama probes, which takes more time and produces verbose output.
### Lesson Learned
* Orchestrating the test suite into 4 separate sequential stages via `scripts/run_suite.py` lets us target specific layers of the application during development:
  * **Stage 1**: Fast Unit/Core (under 1s)
  * **Stage 2**: Semantic & NLP Ingestion
  * **Stage 3**: Persistent DB / Lifecycle
  * **Stage 4**: Model Probing & Evaluation
* This keeps feedback loops extremely fast and prevents test fatigue.

---

## 6. Mocking Optional/Heavy Python Packages in Tests

### Context
When testing pluggable backends (Llama.cpp, OpenAI SDK, SentenceTransformers) and watcher daemons (Watchdog), the test execution environment did not have these heavy optional packages installed. Using standard `@patch` failed because patch attempts to resolve and inspect the modules.
### Lesson Learned
* Prior to test execution or module imports, dummy mocks can be injected directly into `sys.modules` (e.g. `sys.modules['llama_cpp'] = MagicMock()`).
* This permits modules containing imports like `from llama_cpp import Llama` to load successfully and allows us to verify logic paths and factory constructors cleanly without actually installing the underlying libraries in the test pipeline.

---

## 7. Cypher Operator Compliance in KùzuDB

### Context
When implementing active contradiction and transitive inference loops on KuzuStore, standard Python inequality `!=` was used in `WHERE` clauses. This resulted in `RuntimeError: Parser exception: Unknown operation '!='`.
### Lesson Learned
* KùzuDB enforces strict standard Cypher syntax. Unlike Python or other SQL databases, Cypher does not recognize `!=` as an inequality operator.
* The correct Cypher operator for inequality testing is `<>`. Replacing all occurrences of `!=` with `<>` in the Cypher strings resolved the parse errors instantly.

---

## 8. Testable HTTP Server Design with Request Factories

### Context
We wanted to write unit tests for the MCP HTTP server's new dashboard routes (GET `/` and `/api/graph`). Since the handler class definition was previously nested inside a blocking `serve()` method, it could not be imported or run in unit tests.
### Lesson Learned
* Nesting request handlers inside blocking loops blocks testability.
* Refactoring the server to use a class factory `make_handler(tools)` pulls the `Handler` definition out of the blocking socket server. This lets tests instantiate the handler class directly, attach mock reader/writer files (`wfile`, `rfile`), and test request routing (`do_GET()`, `do_POST()`) cleanly without spawning live socket servers.

---

## 9. Bridging Document Drops and Model Probes in Distillation

### Context
When using `ConsensusEngine` to validate facts extracted from documents, `ModelKnowledgeDistiller` would filter out all facts. This occurred because `ModelKnowledgeDistiller` only includes facts carrying a `model_provenance` block to prevent cross-contamination, but standard document-extracted facts lack this metadata.
### Lesson Learned
* Bridging different fact sources is crucial for unified pipeline operations.
* Injecting a synthetic `model_provenance` metadata block mapping the extraction engine (e.g. `svo`, `spacy`) to the fact before passing it to `ModelKnowledgeDistiller` resolves the issue. This allows standard document-extracted drops to be clustered, deduplicated, and ranked using the exact same consensus rules as model-probe drops.

---

## 10. KùzuDB Database Lock and File-vs-Directory Cleanups

### Context
In automated scripts and test cleanups, invoking `shutil.rmtree` on a KùzuDB database path often failed with `NotADirectoryError: [Errno 20] Not a directory` because KùzuDB sometimes creates a flat file instead of a directory structure depending on the environment and active locks.
### Lesson Learned
* Startup/cleanup utilities must handle graph database paths dynamically.
* Wrapping database cleanups with a check (`os.path.isdir(path)`) to selectively run `shutil.rmtree` or `os.remove` prevents execution errors and ensures clean teardown.

---

## 11. Model Weight Entropy Profiling via Logprobs

### Context
When recursively crawling a model's latent weights, models can generate hallucinations or highly uncertain statements. We needed a statistical method to filter these out.
### Lesson Learned
* Requesting token logprobs in GGUF completions (via `logprobs=True`) and remote APIs, and calculating the mathematical average log-probability of the generated text, provides a direct confidence score.
* Filtering out generated facts whose average logprobs fall below a strict threshold (e.g. `-1.5`) successfully prunes highly uncertain or hallucinated factual branches from the graph.

---

## 12. Zero-GitHub-Actions-Minutes Stratified Local CI, Maker/Checker Governance & Local-Path Hygiene

### Context
As `knowledge_graph_pkg` grew across document distillation (`SVOExtractor`), `ModelReduce`, `Watcher` daemons, `KùzuDB`/`Neo4jStore`, and `Graph-RAG` surfaces, keeping CI fast, catching version/license drift, and preventing direct pushes to `main` or leaked local paths required unified mechanical guardrails.
### Lesson Learned
* Restricting `.github/workflows/*.yml` (`ci.yml`, `pr-governance.yml`, `release.yml`, `skill-validation.yml`) to `workflow_dispatch` eliminates automatic CI minute burn while preserving full local parity via `Makefile` (`make canary`, `make skill`, `make lint`, `make local-ci`, `make audit-prs`).
* Mirroring `AGENTS.md` and `.agents/AGENTS.md` (verified byte-identical in `tests/meta/test_governance_and_invariants.py`) alongside `.agents/hooks.json` (`scripts/hooks/pre_pr_safety_gate.py`) and `scripts/github_pr_helper.py` blocks direct `main` pushes and enforces meta-gates before PR creation or merge.
* Mechanical invariant tests catch subtle repository drift immediately: truncated `LICENSE` files, `0.2.0` vs `0.3.0` fallback mismatches in `knowledge_graph_pkg/__init__.py` against `CITATION.cff` and `pyproject.toml`, stray `.db` / `.pyc` artifacts, and machine-local workstation paths in deployment manifests and planning docs.

---

## 13. Real Integration Tests vs. Unspecced Mocks & Store Contract Parity

### Context
During the `v0.4.0` full-codebase audit, unit tests in `tests/test_watcher.py` and `tests/test_neo4j_store.py` passed only because mocks returned fabricated keys (`"status": "success"` + list `"facts"`, or `"statement"`/`"reliability"`) that real producers (`factory.batch_drop`, `ModelKnowledgeDistiller`) never emitted — leaving `WatcherDaemon --graph-db` dead in production and `Neo4jStore` storing empty statements.
### Lesson Learned
* Always keep at least one end-to-end integration test exercising the real producer -> store -> consumer pipeline (`batch_drop` -> `WatcherDaemon` -> `KuzuStore` -> `OntologyDistiller`) inside `tempfile.TemporaryDirectory()` rather than mocking every stage boundary.
* Centralize record normalization (`normalize_fact_record` in `knowledge_graph_pkg/graph_store_base.py`) so `KuzuStore` and `Neo4jStore` accept identical canonical and legacy fact fields and perform idempotent `MERGE ... SET` updates.
* Validate HTTP workspace identifiers (`x-workspace-id`), JWT signatures (`MCP_JWT_SECRET` without plaintext bypasses), WebSocket tenant partitioning, and read-only Cypher guards (`GraphTools.graph_query`) mechanically in `tests/test_mcp_server.py` and `tests/test_graph_tool.py`.

---

## 14. Boundary-Aware Alias Resolution & NetworkX PageRank Fallback

### Context
During Phase 2 review (`entity_resolution.py`, `rag.py`), two subtle failure modes surfaced:
1. Substring matching inside `_jaccard_similarity()` treated any arbitrary substring (`"AI"` in `"Brain Activation"`, `"Fe"` in `"Infection"`) as `1.0`, while deleting the entire graph and rewriting `RELATED` edges away.
2. In NetworkX 3.x without `scipy` installed, `nx.pagerank(graph)` raises `ModuleNotFoundError: No module named 'scipy'`, which a broad `except Exception:` silently flattened to uniform `1/N` scores across all facts; meanwhile PageRank caching ignored edge additions when node counts stayed constant.

### Lesson Learned
* Restrict high-confidence `1.0` entity matching to exact normalized matches, parenthetical acronyms/initials (`"Adenosine Triphosphate (ATP)"` <-> `"ATP"`, `"Large Language Model"` <-> `"LLM"`), and long single-character chemical prefixes (`"D-Glucose"` <-> `"Glucose"`), and rewrite statements on word boundaries after writing merged nodes and remapping `RELATED` edges.
* Track `(store.count(), store.edge_count())` for PageRank cache invalidation, and fall back to pure-Python power iteration when `nx.pagerank()` raises `ImportError` so Graph-RAG ranking stays accurate even in minimal installs without `scipy`.

---

## 15. Unspecced Mocks Masking Backend Contract Drift & Consensus Hash Collisions

### Context
During Phase 3 (`critique.py`, `model_probe.py`, `consensus.py`, `model_eval.py`), unspecced `MagicMock()` backends allowed `FactCritic` to call nonexistent `generate_structured_json`/`generate_text` methods while production `OllamaBackend`/`OpenAICompatibleBackend` only implemented `generate_structured`/`generate_text_with_logprobs`. Meanwhile, `ConsensusEngine` hashed only raw file text across engines so engines 2..N were skipped as duplicates, and `model_eval.py` mixed shard true-positives with gold false-negatives in `tp / (tp + fn)`.

### Lesson Learned
* Use `MagicMock(spec=...)` or explicit fake classes matching the real backend protocol (`generate_structured`, `generate_text_with_logprobs`), and keep at least one non-mocked integration test per multi-stage pipeline (`ConsensusEngine` -> `KnowledgeStore` -> `KuzuStore`).
* Fail closed (`is_factual=False`) on critique parsing/backend errors and validate boolean strings explicitly instead of relying on Python's truthiness (`bool("false") is True`).
* Measure dataset recall against matched gold items (`matched_gold / len(positives)`), not shard prediction counts, so duplicate extracted triples cannot inflate recall past true gold coverage.

---

## 16. Serialization Round-Trips, Capped Similarity Math & Method-Level LRU Cache Traps

### Context
During Phase 4 (`store.py`, `lifecycle.py`, `semantic.py`, `enhanced.py`, `utils.py`, `blockchain.py`, `sharding.py`, `visualization.py`), multiple serialization and stateful bugs were uncovered:
1. Producer/consumer key drift: `store.audit_summary()` expected legacy `"reliability"`/`"statement"` keys while canonical drops wrote `"reliability_rating"`/`"fact_statement"`, whereas `EnhancedKnowledgeGraph.load_from_file()`, `utils.import_from_json()`, and `BlockchainKnowledgeGraph.verify_fact()` failed on `KnowledgeGraph`'s own output (`quality_score` kwarg, comma-string `tags`, `Enum` serialization).
2. `SemanticKnowledgeGraph._calculate_semantic_similarity()` multiplied cosine dot products by `0.405` without scaling magnitudes, capping similarity at `0.405` and causing `find_semantically_similar_facts(threshold=0.5)` to return `[]`.
3. `@lru_cache` on `EnhancedKnowledgeGraph` instance methods pinned `self` references, ignored `cache_enabled=False`, and became stale when mutations skipped `cache_clear()`.
4. `reextract_store()` re-extracted its own `-reextract-` drops on repeat runs, multiplying drops exponentially, and `ShardedKnowledgeGraph.optimize_shards()` cleared memory without deleting stale `.json` files from disk.

### Lesson Learned
* Always test full round-trips across every storage/export/import boundary (`save` -> `load`, `export` -> `import`, `add_fact` -> `verify_fact`, `reextract_store` x 2, `optimize_shards` -> reload) in `tmp_path` so schema and attribute conversions stay symmetric.
* Prefer explicit per-instance dict caches cleared unconditionally on writes over `@lru_cache` on mutable stateful classes.
* When writing generator-style corpus maintenance (`reextract_store`, `optimize_shards`), explicitly skip or replace derived artifacts so repeat runs remain idempotent.

---

## 17. Packaging CLI Subcommands, Lazy Feature Discovery & Zero-Leftover Examples

### Context
During Phase 5 (`cli.py`, `scripts/*.py`, `examples/*.py`, `tests/conftest.py`), several packaging and test-hygiene gaps emerged:
1. Importing optional modules via `importlib.import_module()` inside `tests/conftest.py` eagerly triggered third-party deprecation warnings at pytest collection time, while CLI subcommands calling `get_graph_store()` outside `try ... except ImportError:` masked missing `[graph]` extras with unhandled tracebacks.
2. Calling `distiller.stats()` after `select_facts()` and `retriever.format_context()` after `retriever.retrieve()` ran extraction/search twice per CLI command, and falsy `any([..., args.min_quality])` bypassed `--min-quality 0`.
3. Example scripts and `scripts/test_drive.py` wrote `output/` and store artifacts into the repository working directory unless guarded by temporary directories.

### Lesson Learned
* Use `importlib.util.find_spec()` in pytest fixture probing (`tests/conftest.py`) so skipping missing extras has zero import side effects.
* Wrap both optional module imports AND factory/store constructors inside `try ... except ImportError` in CLI subcommands, and always check `is not None` for numeric CLI filters (`0` is valid!).
* Default demo/example scripts and CLI smoke scripts to `tempfile.TemporaryDirectory()` (with an explicit optional `--output-dir` override) and `Agg` Matplotlib backend so running every example end-to-end leaves `git status` clean.

---

## 18. Editable Package Metadata Refresh, Stale `.egg-info` Shadowing & Minimal-Env Coverage Gates

### Context
During the `v0.4.0` release bump (`PR #7`), `pyproject.toml` was updated from `0.3.0` to `0.4.0` and refreshed offline in `.venv` via `uv pip install --no-deps --no-build-isolation -e .`. However, importing `knowledge_graph_pkg` from the repository root still reported `__version__ == "0.3.0"` because a gitignored `knowledgereduce.egg-info/` directory at the repo root appeared earlier on `sys.path` than `.venv`'s `site-packages/knowledgereduce-0.4.0.dist-info/`. Additionally, full local `.venv` coverage reached `86%+` with `[graph]`, `[viz]`, and FastAPI installed, whereas `.github/workflows/ci.yml` installs minimal `-e ".[dev,hermes]"` (`~78%` coverage).

### Lesson Learned
* After bumping `version` in `pyproject.toml`, always remove stale root `*.egg-info/` directories (`make clean`) AND refresh editable metadata offline (`uv pip install --no-deps --no-build-isolation -e .`) so `importlib.metadata.version("knowledgereduce")` resolves the new version rather than cached `PKG-INFO`.
* Set `--cov-fail-under` according to the minimal CI extra set (`75%` safely covers minimal `.[dev,hermes]` at `~78%` while full local environments run at `86%+`) and keep every governance parity file (`Makefile`, `AGENTS.md`, `.agents/AGENTS.md`, `CONTRIBUTING.md`, `.github/pull_request_template.md`, `.github/workflows/ci.yml`, `tests/meta/test_governance_and_invariants.py`) synchronized.

---

## 19. Offline Golden Regression Benchmarks & Extraction Noise Discovery

### Context
Prior to `PR #8`, `data/gold_set.json` evaluated 15 hand-labeled sentences (`F1 0.857` with `recall >= 0.5`), `data/gold_biochem.json` had no direct test runner, and `data/civic_honors_book.txt` (`139,595` chars) had zero automated pipeline verification. Running a full-book golden benchmark adapted from sibling K-Reduce (`benchmarks/civichonors_book_benchmark.py` and `benchmarks/multisource_factory_benchmark.py`) immediately exposed real extraction flaws: title-case headings (`"Breaking Down"`), question sentences (`?`), single-letter author initials (`"Dr. H. George Frederickson"`), leading-article subjects (`"The university..."`), and discourse/interrogative openers (`"However"`, `"What"`, `"Why"`, `"Many"`).

### Lesson Learned
* End-to-end offline golden benchmarks (`make golden`, `tests/golden/test_civichonors_book_golden.py`, `tests/golden/test_multisource_factory_golden.py`) built on a pure-Python `BaseGraphStore` plus optional `KuzuStore` parity checks catch corpus-scale noise and conflict-reconciliation bugs that small unit fixtures miss — raising `data/gold_set.json` from `F1 0.857` to `1.000` while proving `100%` reference-fact retention, `Hit@1 = 1.0`, and zero heading/question noise across the entire Civic Honors book.




