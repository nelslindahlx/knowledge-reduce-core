---
name: knowledge-reduce-core
description: "Distill documents into compact, reliability-rated, model-absorbable knowledge facts from the local knowledgereduce package. Use when the task involves document distillation, knowledge graphs, model-absorbable JSONL, store/catalog/compile operations, MCP tool invocation, or Hermes wiring for this repo."
version: 0.3.0
author: Nels Lindahl
license: MIT
platforms: [linux, macos, windows]
category: software-development
usage_hint: "Use local Python APIs and CLI subcommands first for offline distillation; fallback to `serve-mcp` HTTP endpoint only when external agents require live MCP network protocol interaction."
required_environment_variables:
  - OPENAI_API_KEY (optional, for openai-based critique)
  - GEMINI_API_KEY (optional, for gemini-based critique)
required_commands:
  - python3
metadata:
  hermes:
    tags: [knowledge-graph, distillation, nlp, training-data, fine-tuning, hermes]
    related_skills: []
---

# KnowledgeReduce Core

Use the installed `knowledge_graph_pkg` package from this repo rather than recreating extraction or distillation logic in scratch scripts.

## Install

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[hermes]"
```

Editable install is preferred in this repo. If `pip install -e .` is blocked by system Python permissions, use a local `.venv` or in-repo paths.

## Optional dependencies

- Ingestion: `.[ingest,pdf]`
- NLP: `.[nlp]`
- Visualization: `.[viz]`
- ModelReduce: `.[model-reduce,graph]`
- Hermes/runtime helpers: `.[hermes]`

---

## Tool surface

### 1. CLI Commands
Exposes the main entrypoint `knowledgereduce` (or `python -m knowledge_graph_pkg`):
* `distill <file>`: Extract SVO triplets from a text source and write to a JSONL training format.
* `drop <file> --store <path>`: Ingest a document, run extraction, and save as a versioned drop shard.
* `audit-store --store <path>`: Scans fact store to audit reliability distributions, empty fields, and duplicate subject clusters.
* `catalog --store <path>`: Search and list facts from the ingestion store.
* `compile --store <path> -o <file>`: Compiles a training dataset shard from all active drops.
* `critique --store <path> --backend <engine>`: Run factual audit verification on extracted facts. Use `--backend none` for offline heuristic critique.
* `serve-mcp`: Launches a FastAPI service providing tool discovery, WebSocket graph streaming, and 3D WebGL dashboard visualization.

### 2. Python API
```python
from knowledge_graph_pkg.store import KnowledgeStore, Drop
from knowledge_graph_pkg.critique import FactCritic
from knowledge_graph_pkg.rag import GraphRAGRetriever
from knowledge_graph_pkg.graph_store_factory import get_graph_store

# Instantiate store
store = KnowledgeStore("store")

# Offline critique fallback
critic = FactCritic(backend_name="none")
report = critic.critique_fact({
    "subject": "Alpha", "predicate": "activates", "object": "Beta",
    "statement": "Alpha activates Beta."
})
```

---

## Preferred invocation

1. **Local Direct API**: Use local python modules (`KnowledgeStore`, `FactCritic`, `get_graph_store`) or local CLI `knowledgereduce` subcommands for repository operations.
2. **HTTP MCP Server**: Invoke `serve-mcp` only when a live network client or external LLM runner needs tool-call gateway access to the graph database.

---

## Safety/timeout defaults

* **Distillation/Crawling Timeouts**: Set bounded CLI limits on crawling loops (`--max-depth 3`, `--max-pages 100`).
* **Critique Throttling**: When using remote API backends, throttle batch critiques to under 5 requests/sec to prevent rate limit blocks.

---

## Hermes prompt patterns

### 1. Running Offline Heuristic Critique
To verify facts without sending data to external APIs:
```bash
knowledgereduce critique --store store --backend none
```

### 2. Ingesting and Auditing a Fact Store
To run a full ingestion sweep followed by a diagnostic audit:
```bash
knowledgereduce drop new_paper.txt --store store --coref
knowledgereduce audit-store --store store
```

---

## Verification & Smoke Sequences

```bash
# Package importability
python3 -c "import knowledge_graph_pkg; print(knowledge_graph_pkg.__version__)"

# Run test suite
python3 -m pytest

# Run local smoke verification
python3 scripts/smoke_skill.py
```
