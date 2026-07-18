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
