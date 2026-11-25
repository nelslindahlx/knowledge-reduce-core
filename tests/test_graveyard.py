"""
Tests for the model graveyard orchestrator (ModelReduce Session 4).
"""

import json
import os

from knowledge_graph_pkg.graveyard import (
    discover_ollama_models,
    run_graveyard,
)


class FakeProber:
    """Stand-in for the probe+store step: records calls, returns fact counts."""

    def __init__(self, facts_per=3, fail_on=None):
        self.calls = []
        self.facts_per = facts_per
        self.fail_on = fail_on or set()

    def __call__(self, model, domain, store, **kw):
        self.calls.append((model, domain))
        if (model, domain) in self.fail_on:
            raise RuntimeError("probe boom")
        return self.facts_per


def test_run_graveyard_probes_all_pairs(tmp_path):
    store_dir = str(tmp_path / "store")
    prober = FakeProber(facts_per=5)
    report = run_graveyard(
        models=["qwen2.5:7b", "phi4:latest"],
        domains=["biochem", "physics"],
        store_dir=store_dir,
        prober=prober,
    )
    assert len(prober.calls) == 4
    assert report.probed == 4
    assert report.total_facts == 20
    assert report.errors == 0


def test_graveyard_resume_skips_completed(tmp_path):
    store_dir = str(tmp_path / "store")
    p1 = FakeProber()
    run_graveyard(["m1"], ["d1", "d2"], store_dir, prober=p1)
    assert len(p1.calls) == 2

    p2 = FakeProber()
    report = run_graveyard(["m1"], ["d1", "d2"], store_dir, prober=p2, resume=True)
    assert len(p2.calls) == 0
    assert report.skipped == 2
    assert report.probed == 0


def test_graveyard_no_resume_reprobes_and_preserves_other_checkpoints(tmp_path):
    store_dir = str(tmp_path / "store")
    run_graveyard(["m1", "m2"], ["d1"], store_dir, prober=FakeProber())

    p2 = FakeProber()
    run_graveyard(["m1"], ["d1"], store_dir, prober=p2, resume=False)
    assert len(p2.calls) == 1

    # Existing checkpoint for m2/d1 must NOT be erased!
    p3 = FakeProber()
    rep3 = run_graveyard(["m2"], ["d1"], store_dir, prober=p3, resume=True)
    assert rep3.skipped == 1
    assert len(p3.calls) == 0


def test_graveyard_does_not_checkpoint_zero_fact_runs(tmp_path):
    store_dir = str(tmp_path / "store")
    rep1 = run_graveyard(["m1"], ["d1"], store_dir, prober=FakeProber(facts_per=0))
    assert rep1.rows[0]["status"] == "empty"

    p2 = FakeProber(facts_per=3)
    rep2 = run_graveyard(["m1"], ["d1"], store_dir, prober=p2, resume=True)
    assert rep2.skipped == 0
    assert rep2.probed == 1
    assert rep2.total_facts == 3


def test_graveyard_records_errors_without_aborting(tmp_path):
    store_dir = str(tmp_path / "store")
    prober = FakeProber(fail_on={("m1", "d2")})
    report = run_graveyard(["m1"], ["d1", "d2", "d3"], store_dir, prober=prober)
    assert len(prober.calls) == 3
    assert report.errors == 1
    assert report.probed == 2


def test_graveyard_report_rows_and_render(tmp_path):
    store_dir = str(tmp_path / "store")
    report = run_graveyard(["m1", "m2"], ["d1"], store_dir, prober=FakeProber(facts_per=4))
    rows = report.rows
    assert len(rows) == 2
    assert all({"model", "domain", "facts", "status"} <= set(r) for r in rows)
    text = report.render()
    assert "m1" in text and "d1" in text


def test_checkpoint_file_written_atomically(tmp_path):
    store_dir = str(tmp_path / "store")
    run_graveyard(["m1"], ["d1"], store_dir, prober=FakeProber())
    ckpt = os.path.join(store_dir, "graveyard_state.json")
    assert os.path.isfile(ckpt)
    assert not os.path.exists(f"{ckpt}.tmp")
    with open(ckpt, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    assert "m1\x00d1" in state["completed"]


def test_discover_ollama_models_parses():
    def fake_lister():
        return ["qwen2.5:7b", "phi4:latest", "mxbai-embed-large:latest"]

    models = discover_ollama_models(lister=fake_lister, exclude_embedding=True)
    assert "qwen2.5:7b" in models and "phi4:latest" in models
    assert all("embed" not in m for m in models)
