# ModelReduce: Session Roadmap

**Goal:** Extend KnowledgeReduce to systematically probe abandoned models, cross-verify their outputs, and distill verified knowledge shards for training new models.

**Philosophy:** Models are ore. KnowledgeReduce is the refinery. Shards are pure metal.

---

## Session 1: Model Probe Infrastructure
**Duration:** 1 session  
**Deliverable:** `knowledge_graph_pkg/model_probe.py` — generic model interrogation engine

### Tasks
1. **Create `ModelProbe` class** supporting multiple backends:
   - HuggingFace (local `transformers` + `accelerate`)
   - vLLM (batched, high-throughput)
   - Ollama (local models already installed)
   - OpenAI-compatible API (remote)
2. **Prompt template system** (`probe_templates.py`):
   - Entity-centric: "State a verified fact about {entity} in {domain}."
   - Relation-centric: "What is the relationship between {e1} and {e2} in {domain}?"
   - Concept-centric: "Explain {concept} in {domain} with key facts."
   - List-centric: "List 5 verified facts about {topic} in {domain}."
   - Negative probes: "What is a common misconception about {topic} in {domain}?"
3. **Seed entity/concept extraction** from existing KnowledgeGraph to bootstrap probes
4. **Batched generation** with configurable temperature, top-p, max tokens
5. **Output schema** — each response becomes a structured "document":
   ```json
   {
     "model": "model-name",
     "backend": "hf|vllm|ollama|api",
     "domain": "biochemistry",
     "prompt_type": "entity|relation|concept|list|negative",
     "prompt": "...",
     "response": "...",
     "metadata": {"temperature": 0.3, "seed": 42, "timestamp": "..."}
   }
   ```

### Verification
```bash
# Quick test with local Ollama model
python -c "
from knowledge_graph_pkg.model_probe import ModelProbe
probe = ModelProbe(backend='ollama', model='qwen2.5:14b')
outputs = probe.probe_domain('biochemistry', n_prompts=10)
print(f'Generated {len(outputs)} outputs')
print(outputs[0])
"
```

### Next Session Input
- Working `ModelProbe` class
- Probe templates for 5+ domains
- Sample outputs from 2+ models

---

## Session 2: Model Output Ingestion & Cross-Model Verification
**Duration:** 1 session  
**Deliverable:** `knowledge_graph_pkg/model_drop.py` + `cross_model.py`

### Tasks
1. **`ModelDrop` class** — ingest model outputs as drops with full provenance:
   - Extends `Drop` with `model_provenance` field
   - Stores prompt, response, model config, generation params
   - Content hash includes model identity (same prompt, different model = different drop)
2. **`CrossModelVerifier`** — run identical probe sets across N models:
   ```python
   verifier = CrossModelVerifier(models=['model-a', 'model-b', 'model-c'])
   results = verifier.probe_domain('physics', n_prompts=500)
   # results: {prompt: [{model, response, facts_extracted}, ...]}
   ```
3. **Agreement detection** — use existing `SVOExtractor` + `KnowledgeDistiller`:
   - Extract facts from each model's response independently
   - Cluster facts by semantic similarity (Jaccard ≥ 0.8)
   - Count distinct models per cluster → reliability signal
4. **Promotion logic** — extend `lifecycle.promote_reliability`:
   - `min_models=2` → LIKELY_TRUE, `min_models=3` → VERIFIED
   - Track which models agreed on each fact

### Verification
```bash
# Run cross-model verification on a domain
python -c "
from knowledge_graph_pkg.cross_model import CrossModelVerifier
verifier = CrossModelVerifier(models=['qwen2.5:14b', 'phi4:latest'])
report = verifier.probe_domain('biochemistry', n_prompts=100)
print(f'Agreement clusters: {report[\"clusters\"]}')
print(f'VERIFIED facts: {report[\"verified\"]}')
print(f'LIKELY_TRUE facts: {report[\"likely_true\"]}')
"
```

### Next Session Input
- Multi-model probing working
- Cross-model agreement clusters
- Facts tagged with model provenance and agreement counts

---

## Session 3: Distillation Pipeline for Model Shards
**Duration:** 1 session  
**Deliverable:** `knowledge_graph_pkg/model_distill.py` + CLI integration

### Tasks
1. **`ModelKnowledgeDistiller`** — specialized distiller for model-derived facts:
   - Inherits `KnowledgeDistiller`, adds model-provenance tracking
   - Filter: `min_model_agreement=2`, `min_reliability=LIKELY_TRUE`
   - Dedup: Jaccard 0.9 within model outputs, 0.85 cross-model
   - Ranking: quality_score × model_agreement_count
2. **Shard output formats** (extend existing serializers):
   ```python
   # SFT chat format with model provenance in metadata
   {"messages": [...], "metadata": {"source_models": ["m1", "m2"], "agreement": 2}}
   
   # IFT format
   {"instruction": "...", "input": "", "output": "...", "metadata": {...}}
   
   # RAG text with citations
   "1. Fact [model-a, model-b; VERIFIED]"
   
   # Model provenance manifest (JSON)
   {"shard": "biochem_v1", "models": [...], "facts": 5000, "verified": 3200, ...}
   ```
3. **Token budgeting** — `--max-tokens` splits across domains proportionally
4. **Train/val split** — stratified by reliability tier and domain

### CLI Integration (`cli.py` additions)
```bash
knowledgereduce model-probe --models qwen2.5:14b,phi4:latest \
  --domains biochem,physics --n-prompts 500 --output ./model_drops/

knowledgereduce model-distill ./model_drops/ --output ./shards/ \
  --format chat,instruction,text --min-agreement 2 \
  --max-tokens 100000 --split 0.9
```

### Verification
```bash
# Full pipeline test
knowledgereduce model-probe --models qwen2.5:14b,phi4:latest \
  --domains biochem --n-prompts 100 --output /tmp/test_drops

knowledgereduce model-distill /tmp/test_drops --output /tmp/test_shards \
  --format chat --min-agreement 2 --max-tokens 5000

# Inspect shard
head -5 /tmp/test_shards/biochem_chat.jsonl
wc -l /tmp/test_shards/biochem_chat.jsonl
```

### Next Session Input
- End-to-end model → shard pipeline working
- Shards in all 3 formats
- Provenance manifest

---

## Session 4: Model Graveyard CLI & Batch Processing
**Duration:** 1 session  
**Deliverable:** `knowledgereduce graveyard` subcommand + batch orchestration

### Tasks
1. **`graveyard` subcommand** — one command eats a directory of models:
   ```bash
   knowledgereduce graveyard /path/to/models/ \
     --domains biochem,physics,law,coding,math \
     --models-per-domain 3 \
     --promote-threshold 2 \
     --output-shards ./shards/ \
     --backend ollama  # or hf, vllm, auto
   ```
2. **Model discovery** — scan directory for:
   - Ollama models (via `ollama list`)
   - HF checkpoints (folders with `config.json`)
   - GGUF files
   - vLLM-served endpoints (config file)
3. **Resource management** — sequential model loading (unload before next), GPU memory cleanup
4. **Resume capability** — checkpoint progress per model per domain
5. **Progress reporting** — rich console output with tables:
   ```
   ┌──────────────┬────────────┬──────────┬──────────┬──────────┐
   │ Model        │ Domain     │ Prompts  │ Facts    │ Verified │
   ├──────────────┼────────────┼──────────┼──────────┼──────────┤
   │ qwen2.5:14b  │ biochem    │ 500      │ 1,247    │ 342      │
   │ phi4:latest  │ biochem    │ 500      │ 1,156    │ 298      │
   │ deepseek:7b  │ biochem    │ 500      │ 1,089    │ 267      │
   ├──────────────┼────────────┼──────────┼──────────┼──────────┤
   │ TOTAL        │ biochem    │ 1,500    │ 2,156    │ 842★     │
   └──────────────┴────────────┴──────────┴──────────┴──────────┘
   ```

### Verification
```bash
# Test with 2 local Ollama models
knowledgereduce graveyard --models qwen2.5:14b,phi4:latest \
  --domains biochem --n-prompts 50 --output-shards /tmp/test_graveyard

# Verify shards produced
ls -la /tmp/test_graveyard/
cat /tmp/test_graveyard/manifest.json
```

### Next Session Input
- `graveyard` command working end-to-end
- Multi-model batch processing
- Rich progress reporting
- Resume/checkpoint capability

---

## Session 5: Evaluation & Quality Gates
**Duration:** 1 session  
**Deliverable:** `knowledge_graph_pkg/model_eval.py` + eval CLI

### Tasks
1. **Gold set construction** — human-verified facts per domain (start with 50-100 per domain)
2. **Evaluator** — run extracted facts against gold:
   ```python
   evaluator = ModelShardEvaluator(gold_set="data/gold_biochem.json")
   metrics = evaluator.evaluate_shard("shards/biochem_chat.jsonl")
   # precision, recall, F1 per reliability tier
   ```
3. **Hallucination rate** — facts claimed by models but contradicted by gold
4. **Coverage** — % of gold facts recovered by model ensemble
5. **Agreement calibration** — does 2-model agreement actually mean 90% precision?
6. **Quality gates** — CI-ready thresholds:
   ```yaml
   # .github/workflows/model-reduce.yml
   gates:
     min_precision_verified: 0.95
     min_recall_likely_true: 0.70
     max_hallucination_rate: 0.05
     min_coverage: 0.60
   ```

### Verification
```bash
knowledgereduce model-eval --shard ./shards/biochem_chat.jsonl \
  --gold ./data/gold_biochem.json --output ./eval_report.json

cat ./eval_report.json | jq '.gates_passed'
```

### Next Session Input
- Evaluation framework with gold sets
- Calibrated agreement thresholds
- CI-ready quality gates

---

## Session 6: Knowledge Graph Query Interface for LLMs
**Duration:** 1 session  
**Deliverable:** `knowledge_graph_pkg/graph_tool.py` + MCP server

### Tasks
1. **Graph query tool** — LLM-callable structured queries:
   ```python
   def graph_query(cypher: str, domain: str = None) -> List[Dict]:
       """Execute Cypher query on KnowledgeGraph."""
   
   def graph_get_fact(fact_id: str) -> Dict:
       """Retrieve full fact with provenance."""
   
   def graph_find_related(fact_id: str, hops: int = 1) -> List[Dict]:
       """Graph traversal from a fact."""
   ```
2. **Natural language → Cypher** — simple template-based translator for common patterns
3. **MCP server** — expose graph tools to any LLM via Model Context Protocol:
   ```bash
   knowledgereduce serve-mcp --store ./store --port 8080
   ```
4. **Tool schema registration** — auto-generate OpenAPI/JSONSchema for LLM function calling

### Verification
```bash
# Start MCP server
knowledgereduce serve-mcp --store ./store &

# Test query from another terminal
curl -X POST localhost:8080/tools/call \
  -d '{"name": "graph_query", "arguments": {"cypher": "MATCH (f) WHERE f.reliability=\"VERIFIED\" RETURN f LIMIT 5"}}'
```

### Next Session Input
- LLM can query the knowledge graph directly
- MCP server running
- Tool schemas documented

---

## Session 7: End-to-End Training Run
**Duration:** 1 session  
**Deliverable:** Trained model on ModelReduce shards + evaluation report

### Tasks
1. **Prepare training data** — compile multi-domain shards:
   ```bash
   knowledgereduce compile --store ./store --output ./training_mix.jsonl \
     --format chat --min-quality 0.7 --reliability LIKELY_TRUE,VERIFIED \
     --max-tokens 200000 --split 0.95
   ```
2. **Run SFT** — using any trainer (LLaMA-Factory, Axolotl, HF SFTTrainer):
   ```bash
   # Example with HF SFTTrainer
   python train_sft.py \
     --model_name_or_path Qwen/Qwen2.5-7B \
     --dataset ./training_mix.jsonl \
     --output_dir ./modelreduce-qwen-7b \
     --lora_rank 32 --learning_rate 2e-4 --num_epochs 3
   ```
3. **Evaluate trained model** — benchmark on:
   - Domain-specific QA (biochem, physics, law, coding)
   - Hallucination rate vs. base model
   - Faithfulness to knowledge graph (query graph → compare answers)
4. **Document results** — comparison table, failure analysis, next steps

### Verification
```bash
# Quick eval with a few test prompts
python eval_model.py --model ./modelreduce-qwen-7b --test-set ./data/test_prompts.jsonl
```

### Next Session Input
- Trained model checkpoint
- Evaluation metrics vs. base model
- Failure cases documented

---

## Session 8: Documentation, Examples, & Release Prep
**Duration:** 1 session  
**Deliverable:** Complete docs, tutorials, PyPI-ready package

### Tasks
1. **README.md** — ModelReduce section with architecture diagram
2. **Tutorial notebook** — `examples/model_reduce_tutorial.ipynb`:
   - Probe 2 models → cross-verify → distill → train LoRA
3. **API docs** — docstrings + `mkdocstrings` config
4. **Dockerfile** — reproducible environment with Ollama + vLLM
5. **GitHub Actions** — CI for model-reduce pipeline (mock models for speed)
6. **PyPI prep** — version bump, changelog, `pip install knowledgereduce[model-reduce]`

---

## Dependency Graph

```
Session 1 (ModelProbe)
    ↓
Session 2 (ModelDrop + CrossModel) ← needs Session 1
    ↓
Session 3 (ModelDistill + CLI) ← needs Session 2
    ↓
Session 4 (Graveyard CLI) ← needs Session 3
    ↓
Session 5 (Evaluation) ← needs Session 3
    ↓
Session 6 (Graph Tools + MCP) ← needs Sessions 1-3
    ↓
Session 7 (Training Run) ← needs Sessions 4, 5, 6
    ↓
Session 8 (Docs + Release) ← needs Session 7
```

---

## Quick-Start Prompt for Next Session

> **Session N Prompt:**
> 
> Continue ModelReduce development from Session {N-1}. 
> 
> **Context:** KnowledgeReduce package at `/Users/nelslindahl/Hermes-Output/knowledgereduce/`. 
> Previous sessions completed: {list}.
> 
> **Current Session Goal:** {session N goal from above}.
> 
> **Deliverable:** {session N deliverable}.
> 
> **Verification:** {session N verification commands}.
> 
> **Key files to modify/create:** {list from session tasks}.
> 
> **Constraints:** Pure Python + networkx + numpy (optional spaCy via `[nlp]` extra). 
> No heavy ML deps in core. Ollama/vLLM/HF backends imported lazily.
> Follow existing code patterns: `extraction.py` for extractors, `distillation.py` for pipeline, `cli.py` for commands.
> 
> Start by reading relevant existing modules, then implement.

---

## Session State Tracker

| Session | Status | Key Output | Next Blocked By |
|---------|--------|------------|-----------------|
| 1: ModelProbe | ✅ Done | `model_probe.py`, `probe_templates.py`, `schemas.py` | — |
| 2: ModelDrop + CrossModel | ✅ Done | `model_drop.py`, `cross_model.py`, `embeddings.py` | 1 |
| 3: ModelDistill + CLI | ✅ Done | `model_distill.py`, CLI extensions | 2 |
| 4: Graveyard CLI | ✅ Done | `graveyard.py`, `graveyard` command | 3 |
| 5: Evaluation | ✅ Done | `model_eval.py`, `data/gold_biochem.json`, `model-eval` CLI | 3 |
| 6: Graph Tools + MCP | ✅ Done | `kuzu_store.py`, `graph_tool.py`, `mcp_server.py`, `graph-ingest`/`serve-mcp` CLI | 1-3 |
| 7: Training Run | ✅ Done (pipeline) | `training_prep.py`, `scripts/train_sft.py`, `model-prep` CLI, `docs/model_reduce_training.md` | 4,5,6 |
| 8: Docs + Release | ✅ Done | README ModelReduce section, `docs/model_reduce_training.md`, v0.3.0 | 7 |

---

## Notes for Future Sessions

- **Always run tests** after each session: `pytest tests/ -q`
- **Commit after each session** with descriptive message: `feat(model-reduce): add ModelProbe with Ollama/HF/vLLM backends`
- **Update `pyproject.toml`** optional dependencies: `model-reduce = ["ollama", "vllm", "accelerate"]`
- **Keep core dependency-free** — lazy import all model backends
- **Reuse existing patterns** — `SVOExtractor`, `KnowledgeDistiller`, `FactQualityFilter`, `KnowledgeStore`
- **Provenance first** — every fact must trace back to model + prompt + generation config

### Session 2 finding (calibration TODO for Session 5)
Live run: qwen2.5:7b + qwen2.5:14b, biochemistry, 4 prompts each → 15 facts,
**0 cross-model clusters** at the default `similarity_threshold=0.8` + exact-SPO
match. Cause: free-form model phrasing diverges ("produce ATP" vs "generate ATP
through cellular respiration"), so Jaccard-on-statements rarely clears 0.8 across
models. The agreement *logic* is correct (unit tests prove clustering + promotion
fire on matching facts). Session 5 must **calibrate** this — options: (a) lower
threshold + measure precision against a gold set, (b) cluster on normalized SPO
(lemmatized subject/object) rather than full statement, (c) embedding similarity
as an opt-in extra. Do NOT hard-lower the default blindly; calibrate against gold.

### Session 3 finding (probe-template wording)
Live `model-probe` → `model-distill` worked end-to-end (2 models, biochemistry,
4 prompts → 2 ModelDrops/3 facts → shard with provenance metadata). But the
`negative`/misconception probe template yields garbled SPO when the model's
"common misconception … correct fact" answer is forced through the SPO schema
(e.g. subject="Biochemistry", predicate="common misconception", object=run-on).
The distill machinery is sound; the issue is upstream in `probe_templates.py`.
TODO (Session 4/5): either give the negative probe its own schema/handling, or
drop it from the default probe mix and keep entity/relation/concept/list.

### Session 5 finding (agreement calibration — HONEST RESULT)
Built `model_eval.py` (ModelShardEvaluator: per-tier precision/recall/F1,
hallucination rate vs gold negatives, coverage, agreement calibration) +
`data/gold_biochem.json` (15 verified + 5 negative) + `model-eval` CLI with
`--ci` gates. Also added embedder support to ModelKnowledgeDistiller so
cross-model clustering is paraphrase-aware (was Jaccard-only → agreement stuck
at 1).

Live calibration, qwen2.5:7b + phi4:latest, biochemistry, 10 prompts each
(90 raw facts → 7 corroborated via embedding clustering @0.82):
  1-model precision = 0.60 (n=5)
  2-model precision = 0.50 (n=2)
  hallucination = 0.286, coverage = 0.267 → quality gates correctly FAILED.

**The 2-model-agreement-implies-higher-precision hypothesis did NOT hold here.**
Honest reasons: (1) n=2 in the agreement bucket is statistically meaningless;
(2) embed threshold 0.82 may cluster non-identical claims → false agreement;
(3) small quantized 7b/phi4 models produce confident-but-wrong biochem that
agreement doesn't filter at low volume. The eval FRAMEWORK is correct and
working — it measured reality instead of assuming. Real calibration needs
MUCH larger probe volume (100s of prompts/model) and likely a tighter embed
threshold (try 0.88-0.90). Do not trust agreement as a precision signal until
re-calibrated at scale. This is the key input for Session 7 (don't train on
this shard as-is).

---

*Generated for Hermes Agent session continuity. Load this file at start of each ModelReduce session.*