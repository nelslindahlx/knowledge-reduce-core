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

## MCP server

```bash
python -m knowledge_graph_pkg serve-mcp --graph-db graph_db --host 127.0.0.1 --port 8080
```

Exposes `GET /tools` and `POST /tools/call` via the dashboard in `mcp_server.py`.
Local assistant/agent use should prefer the local Python APIs first; use the
HTTP server only when an external LLM runtime needs tool-call access.

## Hermes wiring

This repo exposes a canonical skill artifact at:
`.agents/skills/knowledge-reduce-core/SKILL.md`

Install into Hermes skills directory:
```bash
mkdir -p ~/.hermes/skills/software-development/knowledge-reduce-core
cp .agents/skills/knowledge-reduce-core/SKILL.md ~/.hermes/skills/software-development/knowledge-reduce-core/SKILL.md
```

## Repo hygiene

- Do not commit lightweight extras or Hermes-only dependency changes unless needed.
- Restore mutated test artifacts when finished:
  `git checkout -- tests/adaptive_router_q_table.json`
