"""
Tests for LLM-callable graph tools (ModelReduce Session 6).
"""
from unittest.mock import MagicMock

import pytest

from knowledge_graph_pkg.graph_tool import TOOL_SCHEMAS, GraphTools


def _fact(s, p, o, **kw):
    base = {
        "subject": s,
        "predicate": p,
        "object": o,
        "fact_statement": f"{s} {p} {o}.",
        "reliability_rating": "LIKELY_TRUE",
        "cross_model_agreement": 2,
        "source_models": ["m1", "m2"],
        "domain": "biochemistry",
    }
    base.update(kw)
    return base


@pytest.fixture
def tools(tmp_path):
    pytest.importorskip("kuzu")
    from knowledge_graph_pkg.kuzu_store import KuzuStore

    store = KuzuStore(str(tmp_path / "kdb"))
    store.ingest_facts([
        _fact("Mitochondria", "produce", "ATP", reliability_rating="VERIFIED"),
        _fact("Ribosomes", "synthesize", "proteins"),
    ])
    try:
        yield GraphTools(store)
    finally:
        store.close()


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


def test_query_guard_blocks_procedures_semicolons_comments_and_respects_literals():
    mock_store = MagicMock()
    mock_store.query.return_value = [{"i": i} for i in range(10)]
    gt = GraphTools(mock_store, default_limit=3)

    for dangerous in (
        "CALL export_database('/tmp/out')",
        "MATCH (n:Fact) FOREACH (x IN [1] | SET n.x = 1)",
        "EXPORT DATABASE '/tmp/out'",
        "IMPORT DATABASE '/tmp/in'",
        "ATTACH '/tmp/other' AS db2",
        "MATCH (n:Fact) RETURN n; DROP TABLE Fact",
        "MATCH (n:Fact) RETURN n // LIMIT 1",
        "MATCH (n:Fact) /* hide */ RETURN n",
    ):
        with pytest.raises(ValueError):
            gt.graph_query(dangerous)

    # Keywords inside quoted literals must not cause false-positive rejections,
    # and default_limit + Python slice cap must still apply.
    rows = gt.graph_query("MATCH (f:Fact) WHERE f.predicate = 'set limit' RETURN f")
    executed_cypher = mock_store.query.call_args.args[0]
    assert executed_cypher.endswith("LIMIT 3")
    assert len(rows) == 3

    # Even if user specifies higher LIMIT in Cypher, Python-side cap enforces effective limit.
    rows_capped = gt.graph_query("MATCH (f:Fact) RETURN f LIMIT 500")
    assert len(rows_capped) == 3


def test_graph_rag_retrieve_reuses_retriever():
    mock_store = MagicMock()
    mock_store.query.return_value = []
    mock_store.count.return_value = 0
    gt = GraphTools(mock_store)
    gt.graph_rag_retrieve("mitochondria")
    first = gt._retriever
    gt.graph_rag_retrieve("ATP")
    assert gt._retriever is first
