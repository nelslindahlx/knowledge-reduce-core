# KnowledgeReduce — Roadmap v2: The Knowledge Factory

Turn KnowledgeReduce from a one-shot distiller into an **ongoing learning
factory**: every document ingested produces a durable, self-describing
**knowledge drop** (a shard). Drops accumulate in a store; training sets
are compiled *on demand* as reproducible views over that store.

## The shift: split write-side from read-side

```
WRITE SIDE (continuous capture)          READ SIDE (on-demand pull)
  source -> ingest -> extract             store -> query/filter -> compile
        -> distill -> DROP (shard)              -> training set (chat/instruct)
        -> append to store + index              -> versioned dataset snapshot
```

The **store is the durable asset** — not any single output file. Training
sets are reproducible *views* over the store.

## Decisions locked in
- **Store location:** in-repo `store/` (self-contained, version-controlled
  like `data/`). External paths supported later via config.
- **Drop granularity:** one drop **per effort** (per ingestion run), not
  per section.
- **Storage:** pure-Python stdlib — JSONL shards + JSON manifest +
  (Session 9) a SQLite index. No servers.

## What a knowledge drop carries
- facts: statement, triple (subj/pred/obj), Q&A, reliability, quality, tags
- provenance: source URI, content hash, ingest timestamp
- lineage: extractor engine + version, filter settings, coref flag, schema version
- the raw source text (so we can re-extract later when the extractor improves)

Shards are **immutable & append-only**; "edits" produce new versioned drops.

---

## Sessions

**7 — Drop format & sharded store.** ✅ Done. Drop schema + append-only store
(`store/shards/...jsonl` + `store/manifest.json`). Round-trip write/read/validate.

**8 — The factory run.** ✅ Done. `knowledgereduce drop <source>`: run pipeline,
dedup, append a shard, update manifest. Idempotent via content hash. Provenance
stamped.

**9 — Catalog & index.** ✅ Done. SQLite index over all stored facts; query by
source/date/tag/reliability/quality. `knowledgereduce catalog` for store stats.

**10 — Compile on demand.** ✅ Done. `knowledgereduce compile`: assemble a
dataset from the store by reliability/source/category, dedup, token budget,
train/val split. Records which drops it drew from.

**11 — Lifecycle.** ✅ Done. `knowledgereduce lifecycle {promote,contradictions,
reextract}`: re-extract stored sources with a better engine (new drop versions),
reliability promotion across sources, contradiction flagging.

**12 — Automation.** ✅ Done. `knowledgereduce batch <files>|--folder <dir>`:
ingest many sources in one pass, skip unchanged, emit a run report. The
building block for scheduled / watch-folder growth.

## Status: Roadmap v2 complete (Sessions 7-12 shipped, CI-verified).

## Fastest path to payoff
7 -> 8 -> 10 gives the full capture-store-pull loop. 9/11/12 are enrichment.

## Cross-cutting principles
Pure-Python stdlib storage; TDD + CI gate every session; schema versioning
from day one; immutable append-only shards.
