"""
Tests for the graph tool server dispatch and HTTP/WebSocket gateway (ModelReduce Session 6).
"""
from io import BytesIO
import json
import os
import time
from unittest.mock import MagicMock

import pytest

from knowledge_graph_pkg.mcp_server import DASHBOARD_HTML, handle_tool_call, list_tools, make_handler

_TEST_JWT_SECRET = "test-secret-key-at-least-32-bytes-long!"


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
    out = handle_tool_call(
        FakeTools(), "graph_query", {"cypher": "MATCH (f) RETURN f", "limit": 5}
    )
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
    out = handle_tool_call(FakeTools(), "graph_query", {})
    assert out["ok"] is False


def test_handle_non_dict_arguments():
    out = handle_tool_call(FakeTools(), "graph_query", ["bad"])  # type: ignore[arg-type]
    assert out["ok"] is False
    assert "object" in out["error"].lower()


def test_handle_tool_error_is_caught():
    class Boom:
        def graph_query(self, cypher, limit=100):
            raise ValueError("bad cypher")

    out = handle_tool_call(Boom(), "graph_query", {"cypher": "x"})
    assert out["ok"] is False
    assert "bad cypher" in out["error"]


def _make_testable_handler(tools, path="/dashboard", headers=None, body=b""):
    Handler = make_handler(tools)

    class TestableHandler(Handler):
        def __init__(self):
            self.wfile = BytesIO()
            self.rfile = BytesIO(body)
            self.headers = headers or {}
            self.path = path
            self.response_code = 0

        def send_response(self, code):
            self.response_code = code

        def send_header(self, name, value):
            pass

        def end_headers(self):
            pass

        def response_json(self):
            return json.loads(self.wfile.getvalue().decode("utf-8"))

    return TestableHandler()


def test_handler_dashboard_api_and_auth(monkeypatch):
    mock_store = MagicMock()
    mock_store.query.return_value = [{"block_id": "b1", "statement": "Fact statement"}]

    class MockTools:
        store = mock_store

        def graph_query(self, cypher, limit=100):
            return [{"echo": cypher}]

    h = _make_testable_handler(MockTools(), path="/dashboard")
    h.do_GET()
    assert h.response_code == 200
    assert b"<!DOCTYPE html>" in h.wfile.getvalue()

    h = _make_testable_handler(MockTools(), path="/api/graph")
    h.do_GET()
    assert h.response_code == 200
    assert h.response_json()["nodes"][0]["block_id"] == "b1"

    h = _make_testable_handler(MockTools(), path="/invalid-path")
    h.do_GET()
    assert h.response_code == 404

    # Non-dict JSON POST body -> 400
    h = _make_testable_handler(
        MockTools(),
        path="/tools/call",
        headers={"Content-Length": "2"},
        body=b"[]",
    )
    h.do_POST()
    assert h.response_code == 400

    # Oversized POST -> 413
    h = _make_testable_handler(
        MockTools(),
        path="/tools/call",
        headers={"Content-Length": "99999999"},
        body=b"{}",
    )
    h.do_POST()
    assert h.response_code == 413

    # Auth enforcement in stdlib fallback when MCP_JWT_SECRET is set
    jwt = pytest.importorskip("jwt")
    monkeypatch.setenv("MCP_JWT_SECRET", _TEST_JWT_SECRET)
    h = _make_testable_handler(MockTools(), path="/tools")
    h.do_GET()
    assert h.response_code == 401

    token = jwt.encode({"exp": int(time.time()) + 3600}, _TEST_JWT_SECRET, algorithm="HS256")
    h = _make_testable_handler(
        MockTools(), path="/tools", headers={"Authorization": f"Bearer {token}"}
    )
    h.do_GET()
    assert h.response_code == 200


def test_dashboard_html_xss_and_js_hygiene():
    assert "escapeHtml" in DASHBOARD_HTML
    assert "authHeaders" in DASHBOARD_HTML
    assert "data-block-id=" in DASHBOARD_HTML
    assert "edgesDataset" not in DASHBOARD_HTML
    assert 'onclick="pruneFact(' not in DASHBOARD_HTML


def test_fastapi_endpoints(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    jwt = pytest.importorskip("jwt")
    from fastapi.testclient import TestClient

    from knowledge_graph_pkg.mcp_server import make_fastapi_app

    mock_store = MagicMock()
    mock_store.path = str(tmp_path / "graph_db")
    mock_store.count.return_value = 1
    mock_store.query.return_value = [
        {
            "block_id": "b1",
            "statement": "Mitochondria produce ATP.",
            "subject": "Mitochondria",
            "predicate": "produce",
            "object": "ATP",
            "reliability": "VERIFIED",
            "quality": 10,
        }
    ]

    class MockTools:
        store = mock_store

        def graph_query(self, cypher, limit=100):
            return [{"echo": cypher}]

    app = make_fastapi_app(MockTools())
    client = TestClient(app)

    assert client.get("/tools").status_code == 200
    assert b"<!DOCTYPE html>" in client.get("/dashboard").content
    assert "nodes" in client.get("/api/graph").json()
    assert (
        client.post(
            "/tools/call",
            json={"name": "graph_query", "arguments": {"cypher": "MATCH (n) RETURN n"}},
        ).json()["ok"]
        is True
    )
    assert client.post("/tools/call", json=["invalid"]).status_code == 400
    assert client.delete("/api/facts/fact_to_prune").json()["ok"] is True

    # Auth enforcement: reject raw secret, require valid signed JWT with exp, isolate workspaces
    monkeypatch.setenv("MCP_JWT_SECRET", _TEST_JWT_SECRET)
    app_secured = make_fastapi_app(MockTools())
    client_secured = TestClient(app_secured)

    assert client_secured.get("/tools").status_code == 401
    assert (
        client_secured.get(
            "/tools", headers={"Authorization": f"Bearer {_TEST_JWT_SECRET}"}
        ).status_code
        == 401
    )
    assert client_secured.post("/api/notify_update", json={"nodes": []}).status_code == 401

    token_valid = jwt.encode(
        {"exp": int(time.time()) + 3600}, _TEST_JWT_SECRET, algorithm="HS256"
    )
    token_expired = jwt.encode(
        {"exp": int(time.time()) - 3600}, _TEST_JWT_SECRET, algorithm="HS256"
    )
    token_wrong = jwt.encode(
        {"exp": int(time.time()) + 3600},
        "wrong-secret-key-at-least-32-bytes-long!",
        algorithm="HS256",
    )
    auth_hdr = {"Authorization": f"Bearer {token_valid}"}

    assert client_secured.get("/tools", headers=auth_hdr).status_code == 200
    assert (
        client_secured.get("/tools", headers={"Authorization": f"Bearer {token_expired}"}).status_code
        == 401
    )
    assert (
        client_secured.get("/tools", headers={"Authorization": f"Bearer {token_wrong}"}).status_code
        == 401
    )

    # Path traversal / invalid workspace ID blocked with 400
    for bad_ws in ("../evil", "ws/sub", "ws?x=1", "bad ws"):
        resp = client_secured.get(
            "/api/graph", headers={**auth_hdr, "X-Workspace-Id": bad_ws}
        )
        assert resp.status_code == 400

    # Valid tenant workspace creates DB strictly inside tmp_path (never repo root)
    if _has_kuzu():
        headers_ws = {**auth_hdr, "X-Workspace-Id": "my_tenant"}
        assert client_secured.get("/api/graph", headers=headers_ws).status_code == 200
        assert (tmp_path / "graph_db_my_tenant").exists()
        assert not os.path.exists("graph_db_my_tenant")

    # RAG endpoint returns single-pass facts + matching context honoring exclude_unverified
    rag_resp = client_secured.get(
        "/api/rag_retrieve?query=mitochondria&exclude_unverified=false", headers=auth_hdr
    )
    assert rag_resp.status_code == 200
    assert "facts" in rag_resp.json() and "context" in rag_resp.json()
    assert client_secured.get("/api/schema", headers=auth_hdr).status_code == 200

    # Authenticated notify_update succeeds
    notify_resp = client_secured.post(
        "/api/notify_update", headers=auth_hdr, json={"nodes": [], "edges": []}
    )
    assert notify_resp.status_code == 200

    # WebSocket rejects unauthenticated client and accepts valid JWT
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client_secured.websocket_connect("/ws"):
            pass
    assert exc_info.value.code == 1008

    with client_secured.websocket_connect(f"/ws?token={token_valid}&workspace_id=default") as ws:
        resp = client_secured.post(
            "/api/notify_update",
            headers=auth_hdr,
            json={"nodes": [{"block_id": "n1"}], "edges": []},
        )
        assert resp.status_code == 200
        assert resp.json()["recipients"] == 1
        msg = ws.receive_json()
        assert msg["type"] == "graph_update"
        assert msg["nodes"] == [{"block_id": "n1"}]

    for ws_tool in getattr(app_secured.state, "workspace_tools", {}).values():
        ws_store = getattr(ws_tool, "store", None)
        if ws_store is not None and hasattr(ws_store, "close"):
            ws_store.close()


def _has_kuzu() -> bool:
    try:
        import kuzu  # noqa: F401

        return True
    except ImportError:
        return False
