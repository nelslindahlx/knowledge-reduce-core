# ModelReduce: Training Pipeline (Session 7)

The end-to-end path from harvested model knowledge to a fine-tuned LoRA
adapter — and an honest account of what runs where.

## The pipeline

```
probe/graveyard  →  store  →  model-distill  →  model-eval (gate)  →  model-prep  →  train_sft.py
   (Ollama)        (JSONL)      (shard.jsonl)     (PASS/FAIL)         (SFT data)      (LoRA adapter)
```

1. **Harvest** — `knowledgereduce graveyard --models ... --domains ...`
   probes local models into the store (resumable, checkpointed).
2. **Distill** — `knowledgereduce model-distill -o shard.jsonl --min-agreement 2`
   clusters cross-model facts, promotes reliability by agreement, serializes
   to chat-JSONL.
3. **Gate** — `knowledgereduce model-eval --gold <gold>.json --ci`
   scores the shard against a gold set. **This is a hard gate**: a shard that
   fails (low precision / high hallucination) must not be trained on.
4. **Prep** — `knowledgereduce model-prep shard.jsonl -o train.jsonl --min-reliability likely_true --stats`
   validates record shape, drops malformed/empty records, optionally filters
   by reliability, and reports dataset stats. (Pure stdlib; tested in CI.)
5. **Train** — `python scripts/train_sft.py --dataset train.jsonl ...`
   LoRA SFT on a base model. **Not part of the package or CI** (see below).

## Honest hardware reality

The training step (step 5) is deliberately **out of scope for automated
execution** in this project, for two reasons:

1. **Hardware.** A real LoRA fine-tune of a 7B base model needs a capable
   accelerator (CUDA GPU, or Apple silicon with substantial RAM and time).
   It is not feasible on a fanless laptop in any reasonable time, and the
   heavy deps (`torch`, `transformers`, `peft`, `trl`) are intentionally
   *not* in any package extra — install them yourself on training hardware.

2. **Data readiness.** As of the Session 5 calibration, the demo biochem
   shard **fails the quality gates** (hallucination ≈ 0.29; cross-model
   agreement is not yet a reliable precision signal at the volumes tested).
   Training on it would produce a worse model, not a better one. The gate
   exists precisely to stop that.

So `scripts/train_sft.py` is a **complete, parameterized, reviewed** training
script — runnable on the right hardware against a gate-passing shard — but
this repo verifies the *plumbing* (data prep, validation, dataset stats, the
script's dry-run path), not a multi-hour GPU run.

### Validate the plan without training

```bash
python scripts/train_sft.py --dataset train.jsonl --dry-run
```

This loads + validates the dataset and prints the training plan, then exits
before importing any training library.

## What it takes to actually train

1. Probe at **much larger volume** (hundreds of prompts/model, several model
   lineages) so cross-model agreement becomes statistically meaningful.
2. Tighten the embedding match threshold (try ~0.88) and re-run `model-eval`
   until the shard **passes the gates**.
3. On GPU hardware: `pip install torch transformers peft trl datasets accelerate`
   then run `scripts/train_sft.py`.
4. Evaluate the adapter against held-out gold facts before trusting it.
