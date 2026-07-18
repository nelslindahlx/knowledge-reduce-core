"""
Tests for the model graveyard orchestrator (ModelReduce Session 4).

The graveyard batch-probes many models across many domains into a store,
with resume/checkpoint (skip already-probed model+domain pairs) and a run
report. Tests inject a fake prober so no Ollama is needed.
"""
import json
import pytest

from knowledge_graph_pkg.graveyard import (
    run_graveyard, discover_ollama_models, GraveyardReport,
)
from knowledge_graph_pkg.store import KnowledgeStore


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
        # emulate writing a drop: return (n_facts, source_hash)
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
    # 2 models x 2 domains = 4 probes
    assert len(prober.calls) == 4
    assert report.probed == 4
    assert report.total_facts == 20
    assert report.errors == 0


def test_graveyard_resume_skips_completed(tmp_path):
    store_dir = str(tmp_path / "store")
    # First run completes 2 pairs.
    p1 = FakeProber()
    run_graveyard(["m1"], ["d1", "d2"], store_dir, prober=p1)
    assert len(p1.calls) == 2
    # Second run with resume should skip both (checkpoint persisted).
    p2 = FakeProber()
    report = run_graveyard(["m1"], ["d1", "d2"], store_dir, prober=p2, resume=True)
    assert len(p2.calls) == 0
    assert report.skipped == 2
    assert report.probed == 0


def test_graveyard_no_resume_reprobes(tmp_path):
    store_dir = str(tmp_path / "store")
    p1 = FakeProber()
    run_graveyard(["m1"], ["d1"], store_dir, prober=p1)
    p2 = FakeProber()
    run_graveyard(["m1"], ["d1"], store_dir, prober=p2, resume=False)
    assert len(p2.calls) == 1  # re-probed


def test_graveyard_records_errors_without_aborting(tmp_path):
    store_dir = str(tmp_path / "store")
    prober = FakeProber(fail_on={("m1", "d2")})
    report = run_graveyard(["m1"], ["d1", "d2", "d3"], store_dir, prober=prober)
    # all three attempted; one failed but run continued
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


def test_checkpoint_file_written(tmp_path):
    store_dir = str(tmp_path / "store")
    run_graveyard(["m1"], ["d1"], store_dir, prober=FakeProber())
    import os
    ckpt = os.path.join(store_dir, "graveyard_state.json")
    assert os.path.isfile(ckpt)
    state = json.loads(open(ckpt).read())
    assert "m1\x00d1" in state["completed"] or "m1::d1" in str(state)


# ---------- discovery ----------

def test_discover_ollama_models_parses(monkeypatch):
    # Fake `ollama list`-style payload via the injected lister.
    def fake_lister():
        return ["qwen2.5:7b", "phi4:latest", "mxbai-embed-large:latest"]
    models = discover_ollama_models(lister=fake_lister, exclude_embedding=True)
    # embedding model filtered out
    assert "qwen2.5:7b" in models and "phi4:latest" in models
    assert all("embed" not in m for m in models)
