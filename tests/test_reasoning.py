from unittest.mock import patch

import pytest

from knowledge_graph_pkg.cli import main
from knowledge_graph_pkg.kuzu_store import KuzuStore

kuzu = pytest.importorskip("kuzu")


def _fact(
    subject,
    predicate,
    obj,
    reliability="LIKELY_TRUE",
    agreement=2,
    models=None,
    domain="biochemistry",
    quality=5,
):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "fact_statement": f"{subject} {predicate} {obj}.",
        "reliability_rating": reliability,
        "cross_model_agreement": agreement,
        "source_models": models or ["qwen2.5:7b", "phi4:latest"],
        "domain": domain,
        "quality_score": quality,
    }


def test_auto_link_relations(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))
    s.ingest_facts([
        _fact("Mitochondria", "produce", "ATP"),
        _fact("ATP", "is", "energy"),
        _fact("", "unknown", ""),
        _fact("", "still_unknown", "nothing"),
    ])

    n_links = s.auto_link_relations()
    assert n_links == 1

    res = s.query("MATCH (a:Fact)-[r:RELATED]->(b:Fact) RETURN r.predicate AS pred, r.weight AS w")
    assert len(res) == 1
    assert "produce_is" in res[0]["pred"]
    assert res[0]["w"] == 1.0

    # Existing edge weights are preserved on subsequent auto_link calls (ON CREATE SET)
    s.query("MATCH ()-[r:RELATED]->() SET r.weight = 0.75")
    s.auto_link_relations()
    res_after = s.query("MATCH ()-[r:RELATED]->() RETURN r.weight AS w")
    assert res_after[0]["w"] == 0.75
    s.close()


def test_find_contradictions(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))
    s.ingest_facts([
        _fact("Mitochondria", "produce", "ATP", quality=10, reliability="VERIFIED"),
        _fact("Mitochondria", "do not produce", "ATP", quality=2, reliability="POSSIBLY_TRUE"),
        # Unrelated negated predicate must NOT be flagged as a contradiction
        _fact("Mitochondria", "do not destroy", "ATP", quality=8, reliability="LIKELY_TRUE"),
    ])

    contras = s.find_contradictions()
    assert len(contras) == 1
    assert contras[0]["a_pred"] == "produce"
    assert contras[0]["b_pred"] == "do not produce"
    s.close()


def test_find_transitive_inferences(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))
    s.ingest_facts([
        _fact("A", "is", "B", reliability="LIKELY_TRUE"),
        _fact("B", "is", "C", reliability="LIKELY_TRUE"),
        # Self-loop A -> D -> A and UNVERIFIED facts must be excluded
        _fact("C", "is", "A", reliability="UNVERIFIED"),
        _fact("X", "links", "Y", reliability="LIKELY_TRUE"),
        _fact("Y", "links", "X", reliability="LIKELY_TRUE"),
    ])

    s.auto_link_relations()

    infers = s.find_transitive_inferences()
    assert len(infers) == 1
    assert infers[0]["subject"] == "A"
    assert infers[0]["object"] == "C"
    assert infers[0]["predicate"] == "is"
    s.close()


def test_validate_and_reconcile(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))

    # Higher reliability tier wins even if raw numeric quality is lower
    s.ingest_facts([
        _fact("Mitochondria", "produce", "ATP", quality=2, reliability="VERIFIED"),
        _fact("Mitochondria", "do not produce", "ATP", quality=99, reliability="POSSIBLY_TRUE"),
    ])

    result = s.validate_and_reconcile()
    assert len(result["demoted"]) == 1
    assert "do not produce" in result["demoted"][0]["statement"]

    res = s.query("MATCH (f:Fact) WHERE f.predicate = 'do not produce' RETURN f.reliability AS rel")
    assert res[0]["rel"] == "UNVERIFIED"

    res_high = s.query("MATCH (f:Fact) WHERE f.predicate = 'produce' RETURN f.reliability AS rel")
    assert res_high[0]["rel"] == "VERIFIED"
    s.close()


def test_cli_graph_reason(tmp_path, capsys):
    kdb_dir = str(tmp_path / "kdb")
    s = KuzuStore(kdb_dir)
    s.ingest_facts([
        _fact("Mitochondria", "produce", "ATP", quality=10, reliability="VERIFIED"),
        _fact("Mitochondria", "do not produce", "ATP", quality=2, reliability="POSSIBLY_TRUE"),
    ])
    s.close()

    with patch("sys.argv", ["knowledgereduce", "graph-reason", "--graph-db", kdb_dir, "--op", "link"]):
        code = main()
        assert code == 0
        captured = capsys.readouterr()
        assert "created 0 RELATED edges" in captured.out or "created 1 RELATED edges" in captured.out

    with patch("sys.argv", ["knowledgereduce", "graph-reason", "--graph-db", kdb_dir, "--op", "contradictions"]):
        code = main()
        assert code == 0
        captured = capsys.readouterr()
        assert "Conflict between:" in captured.out

    with patch("sys.argv", ["knowledgereduce", "graph-reason", "--graph-db", kdb_dir, "--op", "validate"]):
        code = main()
        assert code == 0
        captured = capsys.readouterr()
        assert "demoted" in captured.out
