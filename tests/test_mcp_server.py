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


def test_handler_dashboard_and_api():
    from knowledge_graph_pkg.mcp_server import make_handler
    from unittest.mock import MagicMock
    import json

    mock_store = MagicMock()
    mock_store.query.return_value = [{"block_id": "b1", "statement": "Fact statement"}]

    class MockTools:
        store = mock_store

    Handler = make_handler(MockTools())

    class TestableHandler(Handler):
        def __init__(self):
            self.wfile = MagicMock()
            self.rfile = MagicMock()
            self.headers = {}
            self.path = "/dashboard"
            self.wfile.write = MagicMock()
            self.response_code = 0

        def send_response(self, code):
            self.response_code = code

        def send_header(self, name, value):
            pass

        def end_headers(self):
            pass

    # 1. Test GET /dashboard
    h = TestableHandler()
    h.do_GET()
    assert h.response_code == 200
    write_args = h.wfile.write.call_args[0][0]
    assert b"<!DOCTYPE html>" in write_args

    # 2. Test GET /api/graph
    h = TestableHandler()
    h.path = "/api/graph"
    h.do_GET()
    assert h.response_code == 200
    write_args = h.wfile.write.call_args[0][0]
    res_data = json.loads(write_args.decode("utf-8"))
    assert "nodes" in res_data
    assert res_data["nodes"][0]["block_id"] == "b1"

    # 3. Test GET invalid path
    h = TestableHandler()
    h.path = "/invalid-path"
    h.do_GET()
    assert h.response_code == 404


def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from knowledge_graph_pkg.mcp_server import make_fastapi_app
    from unittest.mock import MagicMock
    
    mock_store = MagicMock()
    mock_store.query.return_value = [{"block_id": "b1", "statement": "Fact statement"}]
    
    class MockTools:
        store = mock_store
        
        def graph_query(self, cypher, limit=100):
            return [{"echo": cypher}]
            
    app = make_fastapi_app(MockTools())
    client = TestClient(app)
    
    # 1. Test GET /tools
    response = client.get("/tools")
    assert response.status_code == 200
    assert "tools" in response.json()
    
    # 2. Test GET /dashboard
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.content
    
    # 3. Test GET /api/graph
    response = client.get("/api/graph")
    assert response.status_code == 200
    assert "nodes" in response.json()
    
    # 4. Test POST /tools/call
    response = client.post("/tools/call", json={"name": "graph_query", "arguments": {"cypher": "MATCH (n) RETURN n"}})
    assert response.status_code == 200
    assert response.json()["ok"] is True

    # 5. Test DELETE /api/facts/{block_id}
    response = client.delete("/api/facts/fact_to_prune")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert mock_store.query.called

    # 6. Test JWT Token Authentication and Workspace Segregation
    import os
    os.environ["MCP_JWT_SECRET"] = "secret_token_123"
    try:
        app_secured = make_fastapi_app(MockTools())
        client_secured = TestClient(app_secured)
        
        response = client_secured.get("/tools")
        assert response.status_code == 401
        
        headers = {"Authorization": "Bearer secret_token_123"}
        response = client_secured.get("/tools", headers=headers)
        assert response.status_code == 200
        
        headers_ws = {
            "Authorization": "Bearer secret_token_123",
            "X-Workspace-Id": "my_tenant"
        }
        response = client_secured.get("/api/graph", headers=headers_ws)
        assert response.status_code == 200

        # 7. Test GET /api/rag_retrieve
        response = client_secured.get("/api/rag_retrieve?query=mitochondria", headers=headers_ws)
        assert response.status_code == 200
        assert "context" in response.json()
        assert "facts" in response.json()

        # 8. Test GET /api/schema
        response = client_secured.get("/api/schema", headers=headers_ws)
        assert response.status_code == 200
        assert "schema" in response.json()
    finally:
        del os.environ["MCP_JWT_SECRET"]



