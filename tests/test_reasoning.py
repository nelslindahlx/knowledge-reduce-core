import pytest
import os
import sys
from unittest.mock import patch, MagicMock

kuzu = pytest.importorskip("kuzu")

from knowledge_graph_pkg.kuzu_store import KuzuStore
from knowledge_graph_pkg.cli import main

def _fact(subject, predicate, obj, reliability="LIKELY_TRUE", agreement=2,
          models=None, domain="biochemistry", quality=5):
    return {
        "subject": subject, "predicate": predicate, "object": obj,
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
    ])
    
    n_links = s.auto_link_relations()
    assert n_links == 1
    
    # Query relation
    res = s.query("MATCH (a:Fact)-[r:RELATED]->(b:Fact) RETURN r.predicate AS pred")
    assert len(res) == 1
    assert "produce_is" in res[0]["pred"]
    s.close()


def test_find_contradictions(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))
    s.ingest_facts([
        _fact("Mitochondria", "produce", "ATP", quality=10, reliability="VERIFIED"),
        _fact("Mitochondria", "do not produce", "ATP", quality=2, reliability="POSSIBLY_TRUE"),
    ])
    
    contras = s.find_contradictions()
    assert len(contras) == 1
    assert contras[0]["a_pred"] == "produce"
    assert contras[0]["b_pred"] == "do not produce"
    s.close()


def test_find_transitive_inferences(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))
    s.ingest_facts([
        _fact("A", "is", "B"),
        _fact("B", "is", "C"),
    ])
    
    # Must link them first
    s.auto_link_relations()
    
    infers = s.find_transitive_inferences()
    assert len(infers) == 1
    assert infers[0]["subject"] == "A"
    assert infers[0]["object"] == "C"
    assert infers[0]["predicate"] == "is"
    s.close()


def test_validate_and_reconcile(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))
    
    # Fact 1 (High Quality) vs Fact 2 (Low Quality)
    s.ingest_facts([
        _fact("Mitochondria", "produce", "ATP", quality=10, reliability="VERIFIED"),
        _fact("Mitochondria", "do not produce", "ATP", quality=2, reliability="POSSIBLY_TRUE"),
    ])
    
    result = s.validate_and_reconcile()
    assert len(result["demoted"]) == 1
    assert "do not produce" in result["demoted"][0]["statement"]
    
    # Check that it's updated in DB
    res = s.query("MATCH (f:Fact) WHERE f.predicate = 'do not produce' RETURN f.reliability AS rel")
    assert res[0]["rel"] == "UNVERIFIED"
    
    # Check that high quality fact is unaffected
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
    s.close() # Close to release lock

    # Test link CLI operation
    with patch("sys.argv", ["knowledgereduce", "graph-reason", "--graph-db", kdb_dir, "--op", "link"]):
        code = main()
        assert code == 0
        captured = capsys.readouterr()
        assert "created 0 RELATED edges" in captured.out or "created 1 RELATED edges" in captured.out

    # Test contradictions CLI operation
    with patch("sys.argv", ["knowledgereduce", "graph-reason", "--graph-db", kdb_dir, "--op", "contradictions"]):
        code = main()
        assert code == 0
        captured = capsys.readouterr()
        assert "Conflict between:" in captured.out

    # Test validate CLI operation
    with patch("sys.argv", ["knowledgereduce", "graph-reason", "--graph-db", kdb_dir, "--op", "validate"]):
        code = main()
        assert code == 0
        captured = capsys.readouterr()
        assert "demoted" in captured.out
