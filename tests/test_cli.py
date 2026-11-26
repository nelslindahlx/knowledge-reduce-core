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


def test_test_drive_cli(monkeypatch):
    from unittest.mock import patch
    with patch("scripts.test_drive.main") as mock_test_drive:
        with patch("scripts.download_and_crawl.main") as mock_download_and_crawl:
            mock_test_drive.return_value = None
            mock_download_and_crawl.return_value = None
            
            # 1. Base test-drive command
            rc = main(["test-drive"])
            assert rc == 0
            mock_test_drive.assert_called_once()
            mock_download_and_crawl.assert_not_called()
            
            # 2. Test-drive with crawling command
            mock_test_drive.reset_mock()
            rc = main(["test-drive", "--crawl"])
            assert rc == 0
            mock_test_drive.assert_called_once()
            mock_download_and_crawl.assert_called_once()


def test_run_suite_cli(monkeypatch):
    from unittest.mock import patch
    with patch("scripts.run_suite.main") as mock_run_suite:
        mock_run_suite.return_value = None
        rc = main(["run-suite", "--stage", "1", "--verbose"])
        assert rc == 0
        mock_run_suite.assert_called_once_with(["--stage", "1", "--verbose"])


def test_skill_install_cli(monkeypatch):
    from unittest.mock import patch
    with patch("scripts.mirror_skill.main") as mock_mirror:
        with patch("scripts.smoke_skill.run_smoke_tests") as mock_smoke:
            mock_mirror.return_value = 0
            mock_smoke.return_value = 0
            
            # 1. Base install and verify
            rc = main(["skill-install"])
            assert rc == 0
            mock_mirror.assert_called_once()
            mock_smoke.assert_called_once()
            
            # 2. Verify only
            mock_mirror.reset_mock()
            mock_smoke.reset_mock()
            rc = main(["skill-install", "--verify-only"])
            assert rc == 0
            mock_mirror.assert_not_called()
            mock_smoke.assert_called_once()


def test_catalog_min_quality_zero_triggers_query_mode(tmp_path, sample_text, capsys):
    store = tmp_path / "store"
    main(["drop", str(sample_text), "--store", str(store)])
    capsys.readouterr()
    rc = main(["catalog", "--store", str(store), "--min-quality", "0"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "matching facts" in out


def test_invalid_split_returns_exit_code_2(tmp_path, sample_text):
    store = tmp_path / "store"
    main(["drop", str(sample_text), "--store", str(store)])
    assert main(["distill", str(sample_text), "-o", str(tmp_path / "d.jsonl"), "--split", "1.5"]) == 2
    assert main(["compile", "-o", str(tmp_path / "c.jsonl"), "--store", str(store), "--split", "-0.1"]) == 2

    mstore = tmp_path / "mstore"
    _seed_model_store(mstore)
    assert main(["model-distill", "-o", str(tmp_path / "m.jsonl"), "--store", str(mstore), "--split", "2.0"]) == 2


def test_query_graph_calls_retrieve_once_and_supports_json(tmp_path, monkeypatch, capsys):
    from unittest.mock import MagicMock
    mock_store = MagicMock()
    mock_store.count.return_value = 1
    mock_store.query.return_value = [
        {"block_id": "b1", "statement": "Mitochondria produce ATP.", "subject": "Mitochondria",
         "predicate": "produce", "object": "ATP", "reliability": "VERIFIED"}
    ]
    monkeypatch.setattr("knowledge_graph_pkg.graph_store_factory.get_graph_store", lambda path: mock_store)

    # Spy on GraphRAGRetriever.retrieve to confirm single invocation in markdown mode
    from knowledge_graph_pkg.rag import GraphRAGRetriever
    orig_retrieve = GraphRAGRetriever.retrieve
    call_count = 0

    def counting_retrieve(self, *a, **kw):
        nonlocal call_count
        call_count += 1
        return orig_retrieve(self, *a, **kw)

    monkeypatch.setattr(GraphRAGRetriever, "retrieve", counting_retrieve)
    assert main(["query-graph", "Mitochondria", "--graph-db", str(tmp_path / "db")]) == 0
    assert call_count == 1
    assert mock_store.close.call_count == 1

    capsys.readouterr()
    assert main(["query-graph", "Mitochondria", "--graph-db", str(tmp_path / "db"), "--format", "json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed) == 1


def test_critique_missing_store_and_offline_heuristics(tmp_path, sample_text):
    assert main(["critique", "--store", str(tmp_path / "missing"), "--backend", "none"]) == 2

    store = tmp_path / "store"
    main(["drop", str(sample_text), "--store", str(store)])
    assert main(["critique", "--store", str(store), "--backend", "none"]) == 0


def test_batch_missing_file_returns_nonzero(tmp_path):
    rc = main(["batch", str(tmp_path / "does_not_exist.txt"), "--store", str(tmp_path / "store")])
    assert rc == 1


def test_model_prep_and_training_dry_runs(tmp_path, sample_text, capsys):
    # Missing input -> exit 2
    assert main(["model-prep", str(tmp_path / "missing.jsonl"), "-o", str(tmp_path / "out.jsonl")]) == 2

    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}],
                    "metadata": {"reliability": "VERIFIED"}}) + "\n",
        encoding="utf-8",
    )
    prepped = tmp_path / "prepped.jsonl"
    assert main(["model-prep", str(raw), "-o", str(prepped), "--stats"]) == 0
    assert prepped.exists()

    assert main(["train-mlx", "--dataset", str(prepped), "--dry-run"]) == 0
    assert main(["train-sft", "--dataset", str(prepped), "--dry-run"]) == 0

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["train-mlx", "--dataset", str(empty), "--dry-run"]) == 1
    assert main(["train-sft", "--dataset", str(empty), "--dry-run"]) == 1


def test_import_script_missing_returns_3(monkeypatch):
    from knowledge_graph_pkg import cli as cli_mod

    def _boom(_name):
        raise ImportError("no checkout")

    monkeypatch.setattr(cli_mod, "_import_script", _boom)
    assert main(["test-drive"]) == 3
    assert main(["run-suite"]) == 3
    assert main(["skill-install"]) == 3


@pytest.mark.require_kuzu
def test_kuzu_graph_cli_subcommands_end_to_end(tmp_path):
    store_dir = tmp_path / "mstore"
    _seed_model_store(store_dir)
    db_path = str(tmp_path / "kdb")

    assert main(["graph-ingest", "--store", str(store_dir), "--graph-db", db_path, "--min-agreement", "2"]) == 0
    assert main(["graph-reason", "--graph-db", db_path, "--op", "link"]) == 0
    assert main(["graph-reason", "--graph-db", db_path, "--op", "contradictions"]) == 0
    assert main(["graph-reason", "--graph-db", db_path, "--op", "transitive"]) == 0
    assert main(["graph-reason", "--graph-db", db_path, "--op", "validate"]) == 0
    assert main(["resolve-entities", "--graph-db", db_path]) == 0

    inst_out = str(tmp_path / "instructions.jsonl")
    assert main(["compile-graph-instructions", "--graph-db", db_path, "-o", inst_out, "--include-contradictions"]) == 0
    assert Path(inst_out).exists()

    onto_out = str(tmp_path / "ontology.json")
    assert main(["distill-ontology", "--graph-db", db_path, "-o", onto_out]) == 0
    assert Path(onto_out).exists()


def test_run_suite_classify_stages_maps_all_test_files():
    from scripts.run_suite import classify_stages, get_test_files

    all_files = get_test_files()
    stages = classify_stages(all_files)
    mapped_all = [p for paths in stages.values() for p in paths]
    assert len(mapped_all) == len(all_files)
    # Every newly added Cluster B / suite file is explicitly categorized without overlap
    assert "tests/test_blockchain.py" in stages["2"]
    assert "tests/test_cli.py" in stages["4"]
