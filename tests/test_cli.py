"""
Tests for the command-line interface.

The CLI exposes the full pipeline as one command:
    python -m knowledge_graph_pkg distill input.txt -o out.jsonl --format chat

These tests invoke the CLI's main() entrypoint directly with argument
lists, using a temp input file, and assert on exit code + output files.
"""
import json
from pathlib import Path

import pytest

from knowledge_graph_pkg.cli import main


def test_graveyard_missing_domains_errs(tmp_path):
    # no --domains -> argparse error (SystemExit), not a clean run
    with pytest.raises(SystemExit):
        main(["graveyard", "--models", "m1", "--store", str(tmp_path / "s")])


def test_graveyard_runs_with_injected_prober(tmp_path, monkeypatch):
    # Patch the Ollama-backed prober path so no server is needed: replace
    # run_graveyard's prober by patching ModelProbe/OllamaBackend is heavy;
    # instead drive the orchestrator directly to confirm wiring + report.
    from knowledge_graph_pkg.graveyard import run_graveyard

    calls = []

    def fake_prober(model, domain, store, **kw):
        calls.append((model, domain))
        return 2

    report = run_graveyard(["m1", "m2"], ["d1"], str(tmp_path / "store"),
                           prober=fake_prober)
    assert report.probed == 2
    assert report.total_facts == 4
    assert len(calls) == 2


@pytest.fixture
def sample_text(tmp_path):
    p = tmp_path / "src.txt"
    p.write_text(
        "Robert Putnam wrote Bowling Alone. "
        "Marie Curie was born in Warsaw. "
        "She discovered radium."
    )
    return p


def test_distill_chat_writes_jsonl(tmp_path, sample_text):
    out = tmp_path / "train.jsonl"
    rc = main(["distill", str(sample_text), "-o", str(out), "--format", "chat"])
    assert rc == 0
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert lines
    for line in lines:
        rec = json.loads(line)
        assert "messages" in rec


def test_distill_instruction_format(tmp_path, sample_text):
    out = tmp_path / "instruct.jsonl"
    rc = main(["distill", str(sample_text), "-o", str(out), "--format", "instruction"])
    assert rc == 0
    rec = json.loads(out.read_text().splitlines()[0])
    assert "instruction" in rec and "output" in rec


def test_distill_text_format(tmp_path, sample_text):
    out = tmp_path / "digest.txt"
    rc = main(["distill", str(sample_text), "-o", str(out), "--format", "text"])
    assert rc == 0
    assert out.read_text().strip()


def test_coref_flag_attributes_pronoun(tmp_path, sample_text):
    out = tmp_path / "c.jsonl"
    main(["distill", str(sample_text), "-o", str(out), "--format", "chat", "--coref"])
    blob = out.read_text()
    # with coref, "She discovered radium" -> Marie Curie
    assert "Marie Curie" in blob


def test_strict_filter_flag(tmp_path, sample_text):
    out = tmp_path / "s.jsonl"
    rc = main(["distill", str(sample_text), "-o", str(out),
               "--format", "chat", "--filter", "strict"])
    assert rc == 0  # runs cleanly even if strict keeps few/none


def test_missing_input_returns_nonzero(tmp_path):
    out = tmp_path / "x.jsonl"
    rc = main(["distill", str(tmp_path / "nope.txt"), "-o", str(out)])
    assert rc != 0


def test_stats_printed(tmp_path, sample_text, capsys):
    out = tmp_path / "train.jsonl"
    main(["distill", str(sample_text), "-o", str(out), "--format", "chat"])
    captured = capsys.readouterr()
    # CLI reports how many facts/pairs were written
    assert "fact" in captured.out.lower() or "pair" in captured.out.lower()


def test_eval_subcommand_reports_f1(capsys):
    rc = main(["eval", "--gold", "data/gold_set.json"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "f1" in out and "precision" in out and "recall" in out


def test_split_writes_train_and_val(tmp_path, sample_text):
    out = tmp_path / "train.jsonl"
    rc = main(["distill", str(sample_text), "-o", str(out),
               "--format", "chat", "--split", "0.5"])
    assert rc == 0
    assert out.exists()
    assert (tmp_path / "train.jsonl.val").exists()


def test_dedup_store_skips_seen_on_second_run(tmp_path, sample_text):
    out = tmp_path / "t.jsonl"
    store = tmp_path / "seen.json"
    main(["distill", str(sample_text), "-o", str(out),
          "--format", "chat", "--dedup-store", str(store)])
    first = len([l for l in out.read_text().splitlines() if l.strip()])
    assert first > 0
    # Second run with the same store should emit nothing new.
    main(["distill", str(sample_text), "-o", str(out),
          "--format", "chat", "--dedup-store", str(store)])
    second = len([l for l in out.read_text().splitlines() if l.strip()])
    assert second == 0


def test_max_tokens_limits_output(tmp_path, sample_text):
    out = tmp_path / "t.jsonl"
    rc = main(["distill", str(sample_text), "-o", str(out),
               "--format", "chat", "--max-tokens", "5"])
    assert rc == 0
    # tiny budget -> at most a couple of short records
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) <= 2


# ---------- drop subcommand (knowledge factory) ----------

def test_drop_creates_shard_in_store(tmp_path, sample_text):
    store = tmp_path / "store"
    rc = main(["drop", str(sample_text), "--store", str(store)])
    assert rc == 0
    assert (store / "manifest.json").exists()
    import json
    manifest = json.loads((store / "manifest.json").read_text())
    assert len(manifest["drops"]) == 1
    assert manifest["drops"][0]["num_facts"] > 0


def test_drop_is_idempotent_on_unchanged_source(tmp_path, sample_text, capsys):
    store = tmp_path / "store"
    main(["drop", str(sample_text), "--store", str(store)])
    rc = main(["drop", str(sample_text), "--store", str(store)])
    assert rc == 0
    import json
    manifest = json.loads((store / "manifest.json").read_text())
    # second identical run must NOT add a new drop
    assert len(manifest["drops"]) == 1
    assert "skip" in capsys.readouterr().out.lower()


def test_drop_missing_input(tmp_path):
    rc = main(["drop", str(tmp_path / "nope.txt"), "--store", str(tmp_path / "s")])
    assert rc != 0


# ---------- catalog + compile (read side) ----------

def test_catalog_reports_stats(tmp_path, sample_text, capsys):
    store = tmp_path / "store"
    main(["drop", str(sample_text), "--store", str(store)])
    rc = main(["catalog", "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "total facts" in out and "total drops" in out


def test_compile_builds_training_set(tmp_path, sample_text):
    store = tmp_path / "store"
    out = tmp_path / "train.jsonl"
    main(["drop", str(sample_text), "--store", str(store)])
    rc = main(["compile", "-o", str(out), "--store", str(store), "--format", "chat"])
    assert rc == 0
    import json
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert lines
    assert "messages" in json.loads(lines[0])


def test_compile_with_split(tmp_path, sample_text):
    store = tmp_path / "store"
    out = tmp_path / "t.jsonl"
    main(["drop", str(sample_text), "--store", str(store)])
    rc = main(["compile", "-o", str(out), "--store", str(store), "--split", "0.5"])
    assert rc == 0
    assert out.exists() and (tmp_path / "t.jsonl.val").exists()


# ---------- batch + lifecycle (automation + corpus upkeep) ----------

def test_batch_ingests_folder(tmp_path):
    (tmp_path / "a.txt").write_text("Robert Putnam wrote Bowling Alone.")
    (tmp_path / "b.txt").write_text("Marie Curie was born in Warsaw.")
    store = tmp_path / "store"
    rc = main(["batch", "--folder", str(tmp_path), "--store", str(store)])
    assert rc == 0
    import json
    manifest = json.loads((store / "manifest.json").read_text())
    assert len(manifest["drops"]) == 2


def test_lifecycle_contradictions_runs(tmp_path, capsys):
    a = tmp_path / "a.txt"; a.write_text("Robert Putnam wrote Bowling Alone.")
    store = tmp_path / "store"
    main(["drop", str(a), "--store", str(store)])
    rc = main(["lifecycle", "contradictions", "--store", str(store)])
    assert rc == 0
    assert "contradiction" in capsys.readouterr().out.lower()


def test_lifecycle_reextract_runs(tmp_path, capsys):
    a = tmp_path / "a.txt"; a.write_text("Marie Curie discovered radium.")
    store = tmp_path / "store"
    main(["drop", str(a), "--store", str(store)])
    rc = main(["lifecycle", "reextract", "--store", str(store)])
    assert rc == 0
    assert "re-extraction" in capsys.readouterr().out.lower()


# ---------- model-distill (ModelReduce Session 3) ----------

def _seed_model_store(store_dir):
    """Populate a store with two models independently asserting one fact."""
    from knowledge_graph_pkg.store import KnowledgeStore
    from knowledge_graph_pkg.model_drop import ModelDrop

    def po(model, subject, predicate, obj):
        return {"model": model, "backend": "fake", "domain": "biochemistry",
                "prompt": "p", "prompt_type": "entity",
                "structured_response": {"facts": [
                    {"subject": subject, "predicate": predicate, "object": obj,
                     "confidence": 0.9}]}}

    store = KnowledgeStore(str(store_dir))
    store.write_drop(ModelDrop.from_probe_outputs(
        "model-a", "biochemistry", [po("model-a", "Mitochondria", "produce", "ATP")]))
    store.write_drop(ModelDrop.from_probe_outputs(
        "model-b", "biochemistry", [po("model-b", "Mitochondria", "produce", "ATP")]))
    return store


def test_model_distill_chat_writes_corroborated_facts(tmp_path):
    store = tmp_path / "store"
    _seed_model_store(store)
    out = tmp_path / "shard.jsonl"
    rc = main(["model-distill", "-o", str(out), "--store", str(store),
               "--format", "chat", "--min-agreement", "2"])
    assert rc == 0
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["metadata"]["agreement"] == 2
    assert rec["metadata"]["reliability"] == "LIKELY_TRUE"


def test_model_distill_writes_manifest(tmp_path):
    store = tmp_path / "store"
    _seed_model_store(store)
    out = tmp_path / "shard.jsonl"
    manifest = tmp_path / "manifest.json"
    rc = main(["model-distill", "-o", str(out), "--store", str(store),
               "--manifest", str(manifest), "--min-agreement", "2"])
    assert rc == 0
    m = json.loads(manifest.read_text())
    assert m["likely_true"] == 1
    assert set(m["models"]) == {"model-a", "model-b"}


def test_model_distill_missing_store(tmp_path):
    rc = main(["model-distill", "-o", str(tmp_path / "x.jsonl"),
               "--store", str(tmp_path / "nope")])
    assert rc != 0


def test_model_distill_split(tmp_path):
    store = tmp_path / "store"
    _seed_model_store(store)
    out = tmp_path / "s.jsonl"
    rc = main(["model-distill", "-o", str(out), "--store", str(store),
               "--min-agreement", "2", "--split", "0.5"])
    assert rc == 0
    assert out.exists() and (tmp_path / "s.jsonl.val").exists()


def test_query_graph_cli(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    mock_store = MagicMock()
    mock_store.count.return_value = 1
    mock_store.query.return_value = [
        {"block_id": "b1", "statement": "Mitochondria produces ATP", "subject": "Mitochondria", "predicate": "produces", "object": "ATP", "reliability": "VERIFIED"}
    ]
    
    monkeypatch.setattr("knowledge_graph_pkg.graph_store_factory.get_graph_store", lambda path: mock_store)
    
    rc = main(["query-graph", "Mitochondria", "--graph-db", str(tmp_path / "dummy_db")])
    assert rc == 0
