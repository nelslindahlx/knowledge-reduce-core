# KnowledgeReduce — Roadmap

A multi-session plan to evolve KnowledgeReduce from a working heuristic
baseline into a polished, installable, measurable knowledge-distillation
toolkit. Each session ends with something working, tested, and pushed.

## Guiding principles
- **Pure-Python, zero-dependency core stays the default.** Anything heavier
  (spaCy / LLM) is an *opt-in extra*, never a required dependency.
- **TDD throughout** — failing test first, then code, then push.
- **Honest measurement** — every quality claim backed by a number on the
  demo corpus (`data/civic_honors_book.txt`).

---

## Session 1 — Runnable by anyone (packaging + CLI)
Goal: `pip install` it and run the whole pipeline from one command.
- Modern `pyproject.toml` (the old `setup/setup.py` is stale); `pip install -e .`
  works and imports no longer need `PYTHONPATH=.`.
- CLI: `python -m knowledge_graph_pkg distill input.txt -o train.jsonl
  --format chat --filter standard --coref`.
- Tests run after install without the path hack.
- **Deliverable:** one-command pipeline, documented.

## Session 2 — Ingest real-world file formats
Goal: feed it the documents you actually have, not hand-cleaned `.txt`.
- `ingest.py`: HTML→text (generalize the Substack `body_html` extraction),
  plus `.md` and `.txt`.
- PDF support as an opt-in extra (`pip install knowledgereduce[pdf]`, pymupdf).
- **Deliverable:** `create_facts_from_file()` handles `.html`, `.pdf`, `.md`, `.txt`.

## Session 3 — Measure quality (evaluation harness)
Goal: stop eyeballing; get precision/recall numbers.
- Hand-labeled gold set (~40 sentences: expected facts vs. junk).
- `eval.py` scores the extractor (precision/recall/F1) and prints a report.
- Every future change becomes "did F1 go up?" instead of vibes.
- **Deliverable:** `python -m knowledge_graph_pkg eval` with a committed baseline.

## Session 4 — Raise the ceiling (pluggable backends)
Goal: better facts on hard prose without breaking the zero-dependency promise.
- `Extractor` interface; the SVO heuristic becomes one implementation.
- Opt-in spaCy backend (dependency-parse based), gated behind `[nlp]` extra.
- Compare both against the Session 3 gold set; document the tradeoff.
- Core *reduce* stays in our code; this is an optional better engine.
- **Deliverable:** `--engine svo|spacy` with F1 numbers proving the difference.

## Session 5 — Training-data polish
Goal: output that drops straight into a fine-tuning run.
- Train/validation split (`--split 0.9`).
- Cross-run dedup + global fact store (re-running on more docs won't duplicate).
- Token-budget mode ("top facts that fit in N tokens") for RAG context.
- Optional system-prompt templating for chat JSONL.
- **Deliverable:** deduped, budgeted `train.jsonl` + `val.jsonl`.

## Session 6 — Hardening & polish
Goal: behaves like a real OSS package.
- GitHub Actions CI: run the full test suite on every push.
- Coverage report; tighten thin spots.
- README: architecture diagram, badges, contribution notes.
- Tag a `v0.1.0` release.
- **Deliverable:** green CI badge, tagged release.

---

## Sequencing logic
- **1 → 2** first: usability unlocks real testing on real inputs.
- **3 before 4**: need the scorecard before trying to improve quality.
- **5 & 6** are independent polish; can reorder or interleave.

## Fastest path to impact
Sessions **1 + 2 + 3**: usable on real documents *and* a number to chase.
