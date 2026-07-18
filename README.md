# KnowledgeReduce

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Distill documents into compact, reliability-rated, model-absorbable
knowledge facts** — the "reduce" in KnowledgeReduce.

Point it at text (or HTML/Markdown/PDF), and it extracts facts, attributes
pronouns, filters out noise, ranks by quality, deduplicates, and emits
fine-tuning-ready JSONL or a RAG-ready digest. The core is pure-Python and
dependency-light (just `networkx`); spaCy and PDF support are opt-in extras.

```bash
pip install -e .
knowledgereduce distill article.html -o train.jsonl --format chat --coref
```

### Pipeline

```
document (txt/md/html/pdf)
   -> ingest        load_text()
   -> extract       SVOExtractor (default) | spaCy (opt-in)  [+ coreference]
   -> knowledge graph  reliability-rated facts + Q&A
   -> filter        FactQualityFilter (standard | strict)
   -> distill       dedup -> rank by quality -> top_k / token budget
   -> export        chat JSONL | instruction JSONL | text digest  [+ train/val split]
```

Quality is measured, not guessed: `knowledgereduce eval` scores the
extractor against a labeled gold set (current SVO baseline **F1 0.857**).

---

## Advanced Features (knowledge-graph core)

### Core Functionality
- Create and manage knowledge graphs with reliability ratings
- Add, update, and query facts with rich metadata
- Visualize knowledge graphs with customizable layouts
- Import and export knowledge graphs in various formats

### Enhanced Performance
- LRU caching for improved query performance
- Batch operations for efficient data manipulation
- Change tracking and auto-saving capabilities
- Optimized data structures for large knowledge graphs

### Semantic Capabilities
- Entity extraction from unstructured text
- Relationship identification between entities
- Automatic fact creation from text
- Semantic similarity calculation between facts

### Scalability
- Sharding for distributed knowledge graphs
- Efficient handling of very large datasets
- Optimized shard management and balancing
- Cross-shard search and query capabilities

### Vector Embeddings
- Vector-based semantic search
- Fact clustering and categorization
- Query expansion for improved search results
- Similarity matching between facts

### Real-time Streaming
- Event-driven knowledge graph updates
- Real-time data integration
- Streaming data processing
- Event history tracking

### Blockchain Verification
- Immutable fact history
- Blockchain-based verification
- Distributed consensus mechanisms
- Tamper-proof knowledge graphs

## Installation

Core install is pure-Python and dependency-light:

```bash
pip install -e .
```


## Quick Start

```python
from knowledge_graph_pkg import (
    KnowledgeGraph, 
    ReliabilityRating,
    EnhancedKnowledgeGraph,
    VectorKnowledgeGraph,
    StreamingKnowledgeGraph,
    BlockchainKnowledgeGraph
)

# Create an enhanced knowledge graph with caching
kg = EnhancedKnowledgeGraph(cache_enabled=True)

# Add facts with reliability ratings
kg.add_fact(
    fact_id="earth_sun",
    fact_statement="The Earth orbits the Sun",
    category="Astronomy",
    tags=["earth", "sun", "orbit"],
    date_recorded=datetime.now(),
    last_updated=datetime.now(),
    reliability_rating=ReliabilityRating.VERIFIED,
    source_id="astronomy_textbook",
    source_title="Principles of Astronomy",
    author_creator="Dr. Neil Stargazer",
    publication_date=datetime.now(),
    url_reference="https://example.com/astronomy",
    related_facts=[],
    contextual_notes="Fundamental astronomical fact",
    access_level="public",
    usage_count=100
)

# Use vector-based semantic search
vector_kg = VectorKnowledgeGraph(kg)
vector_kg.generate_embeddings()
results = vector_kg.semantic_search("planets orbiting stars")

# Set up real-time streaming
streaming_kg = StreamingKnowledgeGraph(kg)
streaming_kg.add_fact_from_stream({
    'fact_id': 'streaming_fact_1',
    'fact_statement': 'Jupiter has 79 known moons',
    'category': 'Astronomy',
    'tags': ['jupiter', 'moons', 'solar system']
}, source_id='nasa_feed')

# Use blockchain verification
blockchain_kg = BlockchainKnowledgeGraph(kg)
tx_hash = blockchain_kg.add_fact(
    fact_id="blockchain_fact_1",
    fact_statement="Saturn has rings made of ice particles",
    category="Astronomy",
    tags=["saturn", "rings", "solar system"],
    reliability_rating=ReliabilityRating.VERIFIED,
    source_id="astronomy_journal"
)
verification = blockchain_kg.verify_fact("blockchain_fact_1")
```

## Knowledge Factory (ongoing capture → store → compile)

Beyond one-shot distillation, KnowledgeReduce can run as an **ongoing
factory**: every ingestion effort produces a durable, provenance-stamped
**drop** (a shard) appended to a store. Drops accumulate over time;
training sets are **compiled on demand** as reproducible views over the
store. See `docs/ROADMAP-v2.md` for the design.

```bash
# 1. DROP: ingest a source -> one immutable shard per effort (idempotent)
knowledgereduce drop article.html --store store
knowledgereduce drop paper.pdf    --store store   # accumulates

# 2. CATALOG: build a SQLite index and inspect the store
knowledgereduce catalog --store store
knowledgereduce catalog --store store --min-quality 60 --reliability VERIFIED

# 3. COMPILE: assemble a training set as a reproducible view
knowledgereduce compile -o train.jsonl --store store --format chat --split 0.9
knowledgereduce compile -o ctx.jsonl   --store store --max-tokens 2000 --category Science
```

- **Drops are immutable & append-only** — each carries provenance (source,
  content hash, timestamp) and lineage (engine, filter, coref, schema
  version) plus the raw source text, so facts can be re-extracted later
  when the extractor improves.
- **Idempotent**: re-dropping an unchanged source is skipped (content hash).
- **The store is the durable asset**; `train.jsonl` is just a view you can
  regenerate. Compilation records which drops it drew from.

## Installation

```bash
pip install -e .                # core (pure-Python, just networkx)
pip install -e ".[ingest,pdf]"  # + HTML/PDF document ingestion
pip install -e ".[dev]"         # + pytest for development
```

## Command-line usage

After install, distill any document into training data with one command:

```bash
# Chat fine-tuning JSONL, with coreference resolution and quality filtering
knowledgereduce distill input.txt -o train.jsonl --format chat --coref

# Instruction-tuning JSONL
knowledgereduce distill input.txt -o instruct.jsonl --format instruction

# Plain-text ranked digest (RAG context)
knowledgereduce distill input.txt -o digest.txt --format text

# Stricter quality filter (entity subjects only) for entity-rich prose
knowledgereduce distill input.txt -o train.jsonl --filter strict
```

Also runnable as a module: `python -m knowledge_graph_pkg distill ...`.

Flags: `--format {chat,instruction,text}`, `--filter {none,standard,strict}`,
`--coref`, `--engine {svo,spacy}`, `--dedup <0..1>`, `--max-object-len <n>`,
`--min-reliability <level>`.

### Training-data export options

```bash
# Train/validation split -> writes train.jsonl + train.jsonl.val
knowledgereduce distill input.txt -o train.jsonl --split 0.9

# Token budget: keep only the highest-ranked facts that fit ~2000 tokens
knowledgereduce distill input.txt -o ctx.jsonl --max-tokens 2000

# Cross-run dedup: skip facts already emitted in earlier runs
knowledgereduce distill doc1.txt -o out.jsonl --dedup-store seen.json
knowledgereduce distill doc2.txt -o out.jsonl --dedup-store seen.json
```

## Knowledge Distillation (the "reduce" step)

KnowledgeReduce can distill a populated knowledge graph into compact,
high-quality, **model-absorbable** training data. The pipeline is:

```
raw text -> semantic extraction (+ optional coreference)
         -> reliability-rated knowledge graph
         -> distillation: filter -> deduplicate -> rank -> top_k
         -> model-absorbable output (text digest / instruction JSONL / chat JSONL)
```

```python
from knowledge_graph_pkg import (
    KnowledgeGraph, SemanticKnowledgeGraph, KnowledgeDistiller, ReliabilityRating,
)

kg = KnowledgeGraph()
skg = SemanticKnowledgeGraph(kg)

# Extract facts from text. resolve_coref rewrites leading pronouns
# (e.g. "She discovered radium" -> "Marie Curie discovered radium").
skg.create_facts_from_text(
    "Marie Curie was born in Warsaw. She discovered radium.",
    source_id="demo",
    reliability=ReliabilityRating.LIKELY_TRUE,
    resolve_coref=True,
)

# Distill: keep only reliable facts, dedup near-duplicates, rank by quality.
distiller = KnowledgeDistiller(
    kg,
    min_reliability=ReliabilityRating.LIKELY_TRUE,
    dedup_threshold=0.85,
)

print(distiller.to_text())          # ranked plain-text digest (RAG context)
distiller.distill_to_file("train.jsonl", fmt="chat")          # chat SFT JSONL
distiller.distill_to_file("instruct.jsonl", fmt="instruction")  # instruction JSONL
print(distiller.stats())            # {'total_facts', 'selected_facts', 'reduction_ratio', ...}
```

Generated chat records use **real questions** derived from the relation
(via `QAGenerator`), not generic prompts:

```json
{"messages": [{"role": "user", "content": "Where was Marie Curie born?"},
              {"role": "assistant", "content": "Warsaw"}]}
{"messages": [{"role": "user", "content": "What did Marie Curie discover?"},
              {"role": "assistant", "content": "radium"}]}
```

### Ingesting documents from disk

`create_facts_from_file()` (and the CLI) dispatch on file extension:
`.txt` as-is, `.md` strips Markdown, `.html`/`.htm` extracts body text
(including Substack `body_html` JSON), and `.pdf` via the optional
`[pdf]` extra.

```python
skg.create_facts_from_file("source.html", reliability=ReliabilityRating.LIKELY_TRUE)
skg.create_facts_from_file("paper.pdf")   # requires: pip install knowledgereduce[pdf]
```

```bash
knowledgereduce distill article.html -o train.jsonl --format chat
```

## Extraction Quality & Limitations (heuristic baseline)

KnowledgeReduce extraction is **pure-Python and dependency-free** — no
spaCy, no transformers, no LLM. That keeps it fast (the full ~22k-word
demo book processes in well under a second) and runnable anywhere, but it
is fundamentally **heuristic**: it recognizes patterns, it does not
*understand* text. Set expectations accordingly.

### What it does well
- Clean declarative sentences: *"Robert Putnam wrote Bowling Alone."*
- Copula/role: *"Carlsen was President of JCCC."* → `president_of`
- Passive voice: *"The book was published in 2006."* → `published` / `2006`
- Multi-subject: *"Alice and Bob graduated."* → two facts
- Pronoun attribution via opt-in `resolve_coref=True`

### Where it struggles
- **Abstract/argumentative prose** (philosophy, opinion) yields many
  low-quality facts, because the extractor fires on every sentence whether
  or not it states a crisp fact.
- **Possessives and nested clauses** ("in Frederickson's class, where…")
  confuse the subject/object boundaries.
- **No real coreference, no semantic understanding** — a sentence-initial
  word can be mistaken for a subject.

### The quality filter

`FactQualityFilter` removes the obvious junk (stopword subjects, run-on
objects). Two modes, depending on your text:

```python
from knowledge_graph_pkg import FactQualityFilter, KnowledgeDistiller

# Standard: drop stopword subjects + run-on objects. Good general default.
qf = FactQualityFilter(max_object_len=80)

# Strict: also require named-entity subjects + shorter objects.
# Best for entity-rich text (biographies, news); too aggressive for
# abstract prose, where it can over-filter into citations/headers.
strict = FactQualityFilter(max_object_len=60, require_entity_subject=True)

distiller = KnowledgeDistiller(kg, dedup_threshold=0.9, quality_filter=qf)
```

### Measured on the demo book (`data/civic_honors_book.txt`, ~21,765 words)

| Stage | Facts |
|-------|------:|
| Raw extraction | 1,114 |
| After dedup | 1,112 |
| After **standard** filter | **190** |
| After **strict** filter | 13* |

\* On this *abstract* book, strict mode over-filters into the
bibliography — the standard filter (190) is the better tradeoff here. On
entity-rich prose (biographies, news, encyclopedic text) strict mode is
the better choice. **Match the filter to the text.**

### Beyond the heuristic ceiling

To materially improve quality on hard text you would add dependency
parsing (spaCy) or LLM-based extraction. The package ships a **pluggable
extractor interface** so engines are swappable behind one API:

```python
from knowledge_graph_pkg import get_extractor
ex = get_extractor("svo")     # default, pure-Python (always available)
ex = get_extractor("spacy")   # dependency-parse backend, needs [nlp] extra
```

```bash
pip install knowledgereduce[nlp]
python -m spacy download en_core_web_sm
knowledgereduce distill input.txt -o train.jsonl --engine spacy
```

The pure-Python `svo` engine stays the default and the only required path;
`spacy` is opt-in. Score any engine against the gold set with
`knowledgereduce eval` to compare F1.

## Evaluation

Extraction quality is measured against a hand-labeled gold set
(`data/gold_set.json`) — sentences paired with the triples a correct
extractor should produce, including negative examples (headers/fragments
that should yield nothing). Matching is lenient (case-insensitive
substring on subject/object, predicate-family aware).

```bash
knowledgereduce eval                       # uses data/gold_set.json
knowledgereduce eval --gold mygold.json
```

**Current SVO baseline:** precision 0.80, recall 0.92, **F1 0.857**
(15 items). This is the number future extractor changes are measured
against — chase F1, don't eyeball.

## ModelReduce (harvest knowledge from local models)

ModelReduce extends KnowledgeReduce to distill knowledge *out of language
models themselves*: probe local models for structured facts, corroborate
across model lineages, rate reliability by cross-model agreement, and emit
training-ready shards — all Ollama-first and dependency-light.

```bash
pip install -e ".[model-reduce]"       # Ollama-backed probing
pip install -e ".[model-reduce,graph]" # + Kùzu graph store
```

Core package install never needs Ollama:

```bash
pip install -e .
```

Pipeline:

```bash
# 1. Harvest: probe a fleet of models across domains (resumable, checkpointed)
knowledgereduce graveyard --models qwen2.5:7b,phi4:latest \
    --domains biochemistry,physics --n-prompts 20 --store store

# 2. Distill: cluster cross-model facts, promote reliability by agreement
knowledgereduce model-distill -o shard.jsonl --store store --min-agreement 2

# 3. Gate: score against a gold set; fail CI if quality is too low
knowledgereduce model-eval --store store --gold data/gold_biochem.json --embed --ci

# 4. Prep: validate + filter into a training-ready SFT dataset
knowledgereduce model-prep shard.jsonl -o train.jsonl --min-reliability likely_true --stats

# 5. (optional) Query the corpus as a graph
knowledgereduce graph-ingest --store store --graph-db graph_db --embed
knowledgereduce serve-mcp --graph-db graph_db    # LLM-callable tools over HTTP
```

**Reliability ladder** (by distinct-model agreement): 1 → POSSIBLY_TRUE,
2 → LIKELY_TRUE, ≥3 → VERIFIED. Cross-model clustering is paraphrase-aware
when an embedding model (`mxbai-embed-large`) is available, else falls back
to word-overlap.

**Honest caveat:** at small probe volumes, cross-model agreement is *not yet*
a reliable precision signal — the demo biochem shard fails the quality gates
(see `MODELREDUCE_SESSIONS.md`). Probe at scale and re-check the gates before
training. The actual LoRA training run is documented in
`docs/model_reduce_training.md` (it needs real GPU hardware and a
gate-passing shard — by design, not run in CI).

## Hardened Core Enhancements (knowledge-reduce-core)

We have significantly upgraded the core codebase with production-ready features for multi-backend execution, active graph reasoning, glassmorphic visual network analysis, directory-watcher automation, consensus verification, and local training:

### 🔌 Multi-Backend Probing & Offline Embeddings
Probing backends are now fully pluggable. You can probe local GGUF models directly, or connect to remote/local OpenAI-compatible endpoints, using offline sentence-transformer vectors:
```bash
# Probe a local GGUF model via llama-cpp (enforces structured JSON schema)
knowledgereduce model-probe --model models/mistral-7b.gguf --backend llama-cpp

# Probe an OpenAI-compatible API or vLLM endpoint
knowledgereduce model-probe --model qwen --backend openai

# Ingest corroborative facts with offline sentence-transformers
knowledgereduce graph-ingest --store store --graph-db graph_db --embedder sentence-transformers
```

### 🧠 Active Graph Reasoning & Verification
Execute reasoning loops over the Kùzu Graph database to auto-link nodes, detect contradictions, infer transitive shortcuts, and validate reliability:
```bash
# Link matching subjects/objects into RELATED relationships
knowledgereduce graph-reason --op link

# Find contradictory assertions (X vs NOT X)
knowledgereduce graph-reason --op contradictions

# Extract transitive shortcut inferences (A -> B and B -> C implies A -> C)
knowledgereduce graph-reason --op transitive

# Reconcile contradictions, demoting lower-quality conflicting facts to UNVERIFIED
knowledgereduce graph-reason --op validate
```

### 🎨 Glassmorphic Visual Dashboard
Serve LLM-callable tools and a visual dark-mode graph dashboard:
```bash
knowledgereduce serve-mcp --graph-db graph_db --host 127.0.0.1 --port 8080
```
Open `http://localhost:8080/` in any browser to interact with:
* A physics-simulated network graph using `vis-network`.
* Live statistic charts (total facts, domains, avg agreement).
* Concept fact details inspection cards.
* A live Cypher query runner console to execute custom queries and view results in dynamic tables.

### 📁 Real-Time Directory Watcher Daemon
Daemonize the fact-ingestion pipeline to monitor a directory for new document files (`.txt`, `.md`, `.html`, `.pdf`) and automatically ingest them:
```bash
knowledgereduce watch-daemon --dir data/ingest_watch --store store
```
* **Startup Scan**: Automatically scans the folder on boot to ingest any document added while offline.
* **Persistent Logs**: Maintains `watcher_state.db` SQLite database tracking file progress (`PROCESSED`, `SKIPPED`, `FAILED`).
* **Service Deployment**: Plist and systemd configurations are available in `deploy/` for macOS (`launchd`) and Linux (`systemd`).

### ⚖️ Multi-Model Consensus Engine
Ingest a document using SVO and SpaCy engines, compile them into KùzuDB, and run active reconciliation to flag extraction conflicts:
```bash
knowledgereduce consensus input.pdf --engines svo spacy --store store --graph-db graph_db
```

### 🍏 Programmatic Apple Silicon Fine-Tuning
Execute GPU-accelerated local LoRA training programmatically on Apple Silicon:
```bash
knowledgereduce train --model mlx-community/Qwen2.5-3B-Instruct-4bit --data data/train --adapter-path adapters
```

### 🔍 Graph-RAG Hybrid Retriever
Perform hybrid vector/keyword retrieval with adjacent Cypher path traversal from Python:
```python
from knowledge_graph_pkg.kuzu_store import KuzuStore
from knowledge_graph_pkg.rag import GraphRAGRetriever

kstore = KuzuStore("graph_db")
retriever = GraphRAGRetriever(store=kstore, embedder_type="sentence-transformers")

# Format retrieved path contexts directly for injection into LLM prompts
context = retriever.format_context("What does mitochondria generate?", top_k=3)
print(context)
```

### 🕸️ Model Weight Crawling & Entropy Profiling
Recursively crawl a model backend's weights starting from a seed topic, checking log probabilities for confidence-based pruning:
```bash
knowledgereduce crawl --seed "Mitochondria" --backend llama-cpp --model qwen2.5-0.5b-instruct-q4_k_m.gguf --store store --max-depth 2
```

### ⚖️ Critique & Verification Engine (Self-Correction)
Audit facts using a high-capability model backend (like Google Gemini) to identify hallucinations. Flagged facts are automatically demoted to `UNVERIFIED` in KùzuDB:
```bash
knowledgereduce critique --backend gemini --model gemini-1.5-flash --graph-db graph_db
```

### 📦 SFT Dataset Compiler
Compile facts directly into standard instruction-tuning datasets ready for fine-tuning:
```bash
# Export in Alpaca format
knowledgereduce compile-sft -o dataset.json --format alpaca --graph-db graph_db

# Export in ShareGPT format
knowledgereduce compile-sft -o dataset.jsonl --format sharegpt --graph-db graph_db
```


## Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install networkx numpy requests beautifulsoup4 matplotlib pytest
python -m pytest -q          # 40 tests
```

## Examples

See the `examples` directory for detailed usage examples:
- `basic_usage.py`: Simple knowledge graph operations
- `enhanced_features.py`: Advanced features demonstration
- `ultimate_features.py`: Comprehensive example of all capabilities
- `distillation_pipeline.py`: End-to-end text -> facts -> distilled JSONL

## Documentation

For detailed documentation, see the docstrings in the source code or run:

```python
help(knowledge_graph_pkg)
```

## License

MIT
