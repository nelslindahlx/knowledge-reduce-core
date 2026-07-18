"""
Tests for LLM-callable graph tools (ModelReduce Session 6).
"""
import pytest

kuzu = pytest.importorskip("kuzu")

from knowledge_graph_pkg.kuzu_store import KuzuStore
from knowledge_graph_pkg.graph_tool import GraphTools, TOOL_SCHEMAS


def _fact(s, p, o, **kw):
    base = {"subject": s, "predicate": p, "object": o,
            "fact_statement": f"{s} {p} {o}.", "reliability_rating": "LIKELY_TRUE",
            "cross_model_agreement": 2, "source_models": ["m1", "m2"],
            "domain": "biochemistry"}
    base.update(kw)
    return base


@pytest.fixture
def tools(tmp_path):
    store = KuzuStore(str(tmp_path / "kdb"))
    store.ingest_facts([
        _fact("Mitochondria", "produce", "ATP", reliability_rating="VERIFIED"),
        _fact("Ribosomes", "synthesize", "proteins"),
    ])
    return GraphTools(store)


def test_graph_query_tool(tools):
    rows = tools.graph_query("MATCH (f:Fact) RETURN count(f) AS n")
    assert rows[0]["n"] == 2


def test_find_by_subject_tool(tools):
    rows = tools.graph_find_by_subject("Mitochondria")
    assert len(rows) == 1 and rows[0]["object"] == "ATP"


def test_query_rejects_mutations(tools):
    # tool layer should refuse write/DDL Cypher for safety
    with pytest.raises(ValueError):
        tools.graph_query("CREATE (x:Fact {block_id: 'evil'})")
    with pytest.raises(ValueError):
        tools.graph_query("MATCH (f:Fact) DELETE f")


def test_query_auto_limits(tools):
    # a bare MATCH...RETURN with no LIMIT gets one appended
    rows = tools.graph_query("MATCH (f:Fact) RETURN f.subject AS subject", limit=1)
    assert len(rows) == 1


def test_tool_schemas_shape():
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert {"graph_query", "graph_find_by_subject"} <= names
    for t in TOOL_SCHEMAS:
        assert "description" in t and "parameters" in t
