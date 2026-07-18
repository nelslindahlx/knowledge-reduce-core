"""
Tests for the graph tool server dispatch (ModelReduce Session 6).

We test the request-handling logic directly (no socket) so it runs without
kuzu or a live server.
"""
import pytest

from knowledge_graph_pkg.mcp_server import handle_tool_call, list_tools


class FakeTools:
    def graph_query(self, cypher, limit=100):
        return [{"echo": cypher, "limit": limit}]

    def graph_find_by_subject(self, subject):
        return [{"subject": subject}]

    def graph_find_by_reliability(self, reliability, limit=100):
        return [{"reliability": reliability}]


def test_list_tools_returns_schemas():
    tools = list_tools()
    assert any(t["name"] == "graph_query" for t in tools)


def test_handle_graph_query():
    out = handle_tool_call(FakeTools(), "graph_query",
                           {"cypher": "MATCH (f) RETURN f", "limit": 5})
    assert out["ok"] is True
    assert out["result"][0]["echo"].startswith("MATCH")


def test_handle_find_by_subject():
    out = handle_tool_call(FakeTools(), "graph_find_by_subject", {"subject": "ATP"})
    assert out["ok"] is True
    assert out["result"][0]["subject"] == "ATP"


def test_handle_unknown_tool():
    out = handle_tool_call(FakeTools(), "nope", {})
    assert out["ok"] is False
    assert "unknown" in out["error"].lower()


def test_handle_missing_required_arg():
    out = handle_tool_call(FakeTools(), "graph_query", {})  # missing cypher
    assert out["ok"] is False


def test_handle_tool_error_is_caught():
    class Boom:
        def graph_query(self, cypher, limit=100):
            raise ValueError("bad cypher")
    out = handle_tool_call(Boom(), "graph_query", {"cypher": "x"})
    assert out["ok"] is False
    assert "bad cypher" in out["error"]
