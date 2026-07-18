# Contributing to KnowledgeReduce: Developer Playbook & Onboarding Guide

Welcome to the **KnowledgeReduce** contributor guide. This document details how to set up your local development workspace, run the stage-based test suite, verify code style, and maintain high standards.

---

## 🚀 1. Developer Setup & Workspace Initialization

To set up a local development workspace, execute the following:

### Prerequisites
* **Python 3.9+** is required.
* We recommend working within a virtual environment (`venv`).

### Installation
1. **Initialize the Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Upgrade Pip**:
   Modern setuptools features require an updated pip version:
   ```bash
   .venv/bin/python3 -m pip install --upgrade pip
   ```
3. **Install Package in Editable Mode with Extras**:
   Install all development, ingestion, visual, database, and ModelReduce dependencies:
   ```bash
   .venv/bin/pip install -e ".[dev,ingest,pdf,viz,model-reduce,graph]"
   ```
4. **Download SpaCy Model**:
   For NLP-based semantic extraction capabilities, download the default English model:
   ```bash
   .venv/bin/python3 -m spacy download en_core_web_sm
   ```

---

## 🧪 2. Testing Guidelines & Stage-Based Verification

To ensure clean isolation and fast iteration loops, we organize tests into four distinct stages.

### Stage-Based Execution
You can run targeted test suites using the stage manager:

* **Stage 1: Fast Unit & Core Graph / Schemas** (fundamental objects, serialization, schemas)
  ```bash
  .venv/bin/python3 scripts/run_suite.py --stage 1
  ```
* **Stage 2: Semantic & Ingestion / Extraction** (HTML/PDF processing, coreference resolution, spaCy/SVO extraction)
  ```bash
  .venv/bin/python3 scripts/run_suite.py --stage 2
  ```
* **Stage 3: Lifecycle, Database & Graph Integration** (Kùzu store, catalog compilation, SFT training preparation pipelines)
  ```bash
  .venv/bin/python3 scripts/run_suite.py --stage 3
  ```
* **Stage 4: LLM-Dependent ModelReduce Probes & Evaluation** (Ollama integrations, cross-model agreement logic, eval runs)
  ```bash
  .venv/bin/python3 scripts/run_suite.py --stage 4
  ```
* **Run All Stages**:
  ```bash
  .venv/bin/python3 scripts/run_suite.py --stage all
  ```

---

## 🛠️ 3. Pre-Commit Hooks & Quality Gates

We use `pre-commit` to automatically run code formatters and linters before commits are finalized.

### Setup Pre-commit Hooks
1. Install pre-commit in your environment:
   ```bash
   .venv/bin/pip install pre-commit
   ```
2. Register the hooks with Git:
   ```bash
   .venv/bin/pre-commit install
   ```

The hooks will run `black` (for code formatting), `ruff` (for fast linting and import sorting), and general checks on trailing whitespace and YAML syntax.

---

## 📝 4. Code Style & Contribution Standards

To maintain high code quality across the codebase, please adhere to:

* **Type Hinting**: Provide explicit function signature type annotations (`typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Optional`, etc.).
* **Preserve Documentation**: Keep all unrelated docstrings, module documentation, and inline comments unchanged when refactoring.
* **Write Tests**: Every new feature or bug fix must have corresponding tests. Place unit tests under `tests/` prefixed with `test_`.
