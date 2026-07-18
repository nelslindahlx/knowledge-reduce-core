"""
Tests for the KùzuDB graph store (ModelReduce Session 6).

KuzuStore ingests distilled facts as a property graph and answers Cypher
queries + simple lookups. kuzu is an optional dependency (the ``graph``
extra); the whole module is skip-guarded when it isn't installed.
"""
import pytest

kuzu = pytest.importorskip("kuzu")

from knowledge_graph_pkg.kuzu_store import KuzuStore


def _fact(subject, predicate, obj, reliability="LIKELY_TRUE", agreement=2,
          models=None, domain="biochemistry"):
    return {
        "subject": subject, "predicate": predicate, "object": obj,
        "fact_statement": f"{subject} {predicate} {obj}.",
        "reliability_rating": reliability,
        "cross_model_agreement": agreement,
        "source_models": models or ["qwen2.5:7b", "phi4:latest"],
        "domain": domain,
    }


@pytest.fixture
def store(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))
    s.ingest_facts([
        _fact("Mitochondria", "produce", "ATP", reliability="VERIFIED", agreement=3),
        _fact("Ribosomes", "synthesize", "proteins", reliability="LIKELY_TRUE", agreement=2),
        _fact("Glucose", "is", "a monosaccharide", reliability="POSSIBLY_TRUE", agreement=1),
    ])
    return s


def test_ingest_and_count(store):
    assert store.count() == 3


def test_ingest_is_idempotent(tmp_path):
    s = KuzuStore(str(tmp_path / "kdb"))
    f = _fact("Water", "boils at", "100C")
    s.ingest_facts([f])
    s.ingest_facts([f])  # same block_id -> no duplicate
    assert s.count() == 1


def test_query_by_subject(store):
    rows = store.find_by_subject("Mitochondria")
    assert len(rows) == 1
    assert rows[0]["object"] == "ATP"


def test_query_by_reliability(store):
    rows = store.query(
        "MATCH (f:Fact) WHERE f.reliability = 'VERIFIED' RETURN f.statement AS s")
    assert len(rows) == 1
    assert "Mitochondria" in rows[0]["s"]


def test_raw_cypher_count(store):
    rows = store.query("MATCH (f:Fact) RETURN count(f) AS n")
    assert rows[0]["n"] == 3


def test_filter_by_domain(store):
    rows = store.query(
        "MATCH (f:Fact) WHERE f.domain = 'biochemistry' RETURN f.subject AS subj")
    assert len(rows) == 3


def test_reopen_persists(tmp_path):
    path = str(tmp_path / "kdb")
    s1 = KuzuStore(path)
    s1.ingest_facts([_fact("DNA", "stores", "genetic information")])
    s1.close()
    s2 = KuzuStore(path)
    assert s2.count() == 1
