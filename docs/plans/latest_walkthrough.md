# Latest Walkthrough: Phase 3 ModelReduce Probing, Consensus, Critique, Eval, Crawler, Graveyard & Training

Summary of Phase 3 remediation from `docs/plans/master_plan_v0_4_0.md` executed on branch `fix/model-reduce-pipeline`.

## Changes Applied
- **Model Probing & Prompt Templates (`model_probe.py`, `probe_templates.py`)**:
  - Updated `OpenAICompatibleBackend.generate_structured()` to emit `response_format={"type": "json_schema", ...}` for non-empty schemas (and `"json_object"` when empty), added `timeout` across Ollama/OpenAI/Gemini clients, and logged API errors.
  - Sanitized Gemini `response_schema` against `minimum`/`maximum`/`additionalProperties`, updated default Gemini model to `gemini-2.0-flash`, enabled `logits_all=True` and numeric filtering of `None` token logprobs on `LlamaCppBackend`, fixed `get_backend` kwargs routing, varied per-probe seeds and no-entity domain prompts, and guarded `seed_entities_from_graph(..., limit <= 0)`.
- **Fact Critic (`critique.py`)**:
  - Aligned `FactCritic.critique_fact()` with the real `generate_structured(prompt, schema)` and `generate_text_with_logprobs(prompt)` backend contract, enforced standard lowercase `CRITIQUE_SCHEMA`, added strict `_parse_bool()` (so `"false"` parses as `False`), failed closed (`is_factual=False`) on backend/parse errors, added tautology (`subj == obj`) rejection in offline heuristics, and stopped masking invalid backend / API-key `ValueError`s.
- **Consensus, Cross-Model Verification, Distillation & Drops (`factory.py`, `consensus.py`, `cross_model.py`, `model_distill.py`, `model_drop.py`)**:
  - Added `batch_drop(..., include_engine_in_hash=True)` and wired `ConsensusEngine` to stamp real engine provenance (`"svo"`, `"spacy"`), retain drops across multiple extractors, and route graph targets via `get_graph_store()` (supporting Neo4j URIs and KùzuDB).
  - Inverted the buggy `ml in name` model weight check in `cross_model.py` to normalized boundary matching (handling colons and `"gpt-4o"`), guarded empty-SPO clusters and `[]` embeddings, tracked `UNVERIFIED` cluster counts, fixed `min_reliability` validation and `quality_score: None` handling in `ModelKnowledgeDistiller`, preserved valid compound phrases (`"DNA polymerase"`) in `model_drop.py`, and hardened `_safe()` drop IDs against path traversal.
- **Embeddings, Eval, Crawler, Graveyard & Training (`embeddings.py`, `model_eval.py`, `crawler.py`, `graveyard.py`, `training_prep.py`, `train.py`)**:
  - Hardened `cosine_similarity`, bounded thread-safe `EmbeddingCache` (skipping `[]`), and O(1) `VectorIndex` dimension validation.
  - Fixed `ModelShardEvaluator` recall (`matched_gold / len(positives)`), empty-shard `0.0` precision, negation/boundary matching in `_lenient_match`, tier normalization, partial `check_gates` overrides, and numeric report sorting.
  - Propagated missing-engine `ImportError` in `ModelCrawler`, added `max_topics` cap and `None` text guards, stamped string `reliability_rating` + `model_provenance`, closed `graveyard.py` checkpoint handles atomically, preserved unrelated checkpoints when `resume=False`, and skipped checkpointing 0-fact runs.
  - Made `prepare_sft_dataset()` atomic and in-place safe, accepted `"system"` messages, rejected non-string content, added `prepare_mlx_dataset()`, and merged `mlx_lm.lora.CONFIG_DEFAULTS` in `MLXTrainer`.
- **Upgraded Test Coverage**:
  - Expanded 15+ test modules (`test_critique.py`, `test_model_probe.py`, `test_backends.py`, `test_hardening.py`, `test_probe_templates.py`, `test_consensus.py`, `test_cross_model.py`, `test_weighted_consensus.py`, `test_model_distill.py`, `test_model_drop.py`, `test_embeddings.py`, `test_model_eval.py`, `test_crawler.py`, `test_graveyard.py`, `test_training_prep.py`, `test_train_wrapper.py`), eliminating permanent `sys.modules` leaks in `test_train_wrapper.py` and `test_model_probe.py`.

## Verification Results

| Check | Result | Timing |
|---|---|---|
| `make local-ci` (`canary`, `skill`, `lint`, `test`) | `395 passed, 5 skipped`, `72.42%` coverage (`>= 58%` gate) | `5.60s` (`pytest`) |
| `make smoke-pr` (`canary`, `skill`, `lint`, Stage 1 & 2) | Stage 1 & Stage 2 passed cleanly (`1` optional skip) | Passed |
| `make canary` (`-W error::Warning` import + `tests/meta/`) | `10 passed`, zero import warnings | `< 1s` |
| `make lint` (`ruff check` + strict governance check/format) | `All checks passed!`, `4 files already formatted` | `< 1s` |
