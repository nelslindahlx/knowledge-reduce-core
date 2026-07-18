---
name: knowledge-reduce-core
description: "Distill documents into compact, reliability-rated, model-absorbable knowledge facts from the local knowledgereduce package. Use when the task involves document distillation, knowledge graphs, model-absorbable JSONL, store/catalog/compile operations, MCP tool invocation, or Hermes wiring for this repo."
version: 0.3.0
author: Nels Lindahl
license: MIT
platforms: [linux, macos, windows]
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

Editable install is preferred in this repo. If `pip install -e .` is blocked by
system Python permissions, use a local `.venv` or in-repo paths.

## Optional dependencies

- Ingestion: `.[ingest,pdf]`
- NLP: `.[nlp]`
- Visualization: `.[viz]`
- ModelReduce: `.[model-reduce,graph]`
- Hermes/runtime helpers: `.[hermes]`

## Run

CLI entrypoint:
```bash
knowledgereduce --help
knowledgereduce distill input.txt -o train.jsonl --format chat --coref
knowledgereduce drop article.html --store store
knowledgereduce catalog --store store
knowledgereduce compile -o train.jsonl --store store --format chat --split 0.9
knowledgereduce serve-mcp --graph-db graph_db --host 127.0.0.1 --port 8080
```

Module entrypoint:
```bash
python -m knowledge_graph_pkg distill input.txt -o train.jsonl --format chat
```

## Canonical API patterns

### 1. Extract and distill a small store
```python
from datetime import datetime
from knowledge_graph_pkg import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.semantic import SemanticKnowledgeGraph
from knowledge_graph_pkg.distillation import KnowledgeDistiller

text = """Marie Curie was born in Warsaw. She discovered radium with Pierre Curie."""

kg = KnowledgeGraph()
skg = SemanticKnowledgeGraph(kg)
created = skg.create_facts_from_text(
    text,
    source_id='curie_demo',
    reliability=ReliabilityRating.UNVERIFIED,
    resolve_coref=True,
    use_svo=True,
)

d = KnowledgeDistiller(
    kg,
    min_reliability=ReliabilityRating.UNVERIFIED,
    dedup_threshold=0.0,
    top_k=10,
)

selected = d.select_facts()
print('count=', len(selected))
for f in selected:
    print(f.get('fact_statement'))
print(d.to_chat_jsonl())
print(d.to_instruction_jsonl())
print(d.to_text())
```

Observed behavior on the Curie demo:
- created 4 fact IDs
- examples produced:
  - Marie Curie was born in Warsaw.
  - Marie Curie discovered radium with Pierre Curie.
  - Pierre Curie won Nobel Prize in Physics in 1903.
  - Nobel Prize later won a second Nobel in Chemistry in 1911.
- every fact got `quality_score=12` and `reliability_rating=UNVERIFIED`

### 2. Inspect a small store
```python
for node_id, data in kg.graph.nodes(data=True):
    print(node_id, data.get('fact_statement'), data.get('reliability_rating'))
```

### 3. Keep the contract narrow
- Prefer `create_facts_from_text()` with `use_svo=True`
- Treat regex-based extraction as lightweight heuristic QA, not gold QA
- `to_chat_jsonl()` is the safest output path for Hermes
- CLI QA stays at `f1=0.857` on the included gold set

## Hermes prompt patterns

### distill
User: "Distill this text into training JSONL with coref."
Assistant:
```bash
python -m knowledge_graph_pkg distill input.txt -o train.jsonl --format chat --coref
```

### drop
User: "Ingest this article into a fresh store."
Assistant:
```bash
knowledgereduce drop article.html --store store
python -m knowledge_graph_pkg catalog --store store
```

### catalog
User: "Show me what is in the store."
Assistant:
```bash
knowledgereduce catalog --store store
```

### compile
User: "Compile a chat shard from the store."
Assistant:
```bash
knowledgereduce compile -o train.jsonl --store store --format chat --split 0.9
```

### serve-mcp / MCP dispatch
User: "Expose graph tools for another LLM."
Assistant:
```bash
python -m knowledge_graph_pkg serve-mcp --graph-db graph_db --host 127.0.0.1 --port 8080
```

MCP introspection:
```bash
python -c "from knowledge_graph_pkg.mcp_server import list_tools; print(list_tools())"
```

HTTP surface:
- `GET /tools`
- `POST /tools/call`

Local assistant/agent use should prefer the local Python APIs above; use the
HTTP server only when an external LLM runtime needs tool-call access.

## Preferred invocation

1. Local Python APIs first
2. CLI for one-off verification
3. `serve-mcp` only for external runtime tool dispatch

## Verification & Smoke Sequences

```bash
# Package importability
python -c "import knowledge_graph_pkg; print(knowledge_graph_pkg.__version__)"

# Small-store smoke path
python - <<'PY'
from knowledge_graph_pkg import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.semantic import SemanticKnowledgeGraph
from knowledge_graph_pkg.distillation import KnowledgeDistiller

text = "Marie Curie discovered radium with Pierre Curie."
kg = KnowledgeGraph()
skg = SemanticKnowledgeGraph(kg)
skg.create_facts_from_text(text, source_id='smoke', reliability=ReliabilityRating.UNVERIFIED, resolve_coref=True)
d = KnowledgeDistiller(kg, min_reliability=ReliabilityRating.UNVERIFIED, dedup_threshold=0.0, top_k=10)
selected = d.select_facts()
assert len(selected) >= 1
print('smoke ok:', len(selected))
PY

# Core extraction path
python -m knowledge_graph_pkg eval --gold data/gold_set.json

# CLI help
python -m knowledge_graph_pkg --help
```

## Safety/timeout defaults

- extraction smoke: < 1s
- distill/drop/catalog: 30s for small inputs
- eval gold: 60s
- serve-mcp startup: 10s, then treat as long-running service
- batch/graveyard/model jobs: document explicit bounded timeouts before use

## Repo hygiene

- Do not commit lightweight extras or Hermes-only dependency changes unless needed.
- Restore mutated test artifacts when finished:
  `git checkout -- tests/adaptive_router_q_table.json`

## Hermes wiring

Canonical skill artifact:
`.agents/skills/knowledge-reduce-core/SKILL.md`

Install into Hermes:
```bash
python3 scripts/mirror_skill.py
```

Verify install matches canonical source:
```bash
diff -q .agents/skills/knowledge-reduce-core/SKILL.md ~/.hermes/skills/software-development/knowledge-reduce-core/SKILL.md
```
