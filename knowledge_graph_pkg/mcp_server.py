"""
Lightweight tool server for the knowledge graph (ModelReduce Session 6).

Exposes :class:`~knowledge_graph_pkg.graph_tool.GraphTools` over a simple
HTTP+JSON interface so any LLM (or ``curl``) can query the distilled-fact
graph as callable tools. Stdlib-only (``http.server``) -- no extra
dependencies -- so it ships in CI and runs anywhere.

Endpoints:
* ``GET  /tools``            -> the tool schemas (for function-calling setup)
* ``POST /tools/call``       -> ``{"name": ..., "arguments": {...}}`` -> result

The dispatch core (:func:`handle_tool_call`, :func:`list_tools`) is separated
from the socket layer so it is unit-testable without binding a port.
"""

import json
from typing import Any, Dict

from .graph_tool import TOOL_SCHEMAS

# Required arguments per tool (mirrors TOOL_SCHEMAS "required").
_REQUIRED = {t["name"]: t["parameters"].get("required", []) for t in TOOL_SCHEMAS}


def list_tools() -> list:
    """Return the tool schemas (for LLM function-calling / MCP registration)."""
    return TOOL_SCHEMAS


def handle_tool_call(tools: Any, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch one tool call against a GraphTools-like object.

    Returns ``{"ok": True, "result": [...]}`` or ``{"ok": False, "error": ...}``.
    Never raises -- errors are returned as data so the server can always reply.
    """
    method = getattr(tools, name, None)
    if method is None or name not in _REQUIRED:
        return {"ok": False, "error": f"unknown tool: {name}"}
    arguments = arguments or {}
    missing = [r for r in _REQUIRED[name] if r not in arguments]
    if missing:
        return {"ok": False, "error": f"missing required argument(s): {', '.join(missing)}"}
    try:
        result = method(**arguments)
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001 - return errors as data
        return {"ok": False, "error": str(exc)}


def serve(store_path: str, host: str = "127.0.0.1", port: int = 8080) -> None:  # pragma: no cover
    """Start the HTTP tool server backed by a KuzuStore at ``store_path``.

    Blocking call -- runs until interrupted. Imported lazily so the module
    stays importable (and testable) without kuzu installed.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from .kuzu_store import KuzuStore
    from .graph_tool import GraphTools

    tools = GraphTools(KuzuStore(store_path))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/tools":
                self._send(200, {"tools": list_tools()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/tools/call":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"ok": False, "error": "invalid JSON"})
                return
            out = handle_tool_call(tools, req.get("name", ""), req.get("arguments", {}))
            self._send(200 if out.get("ok") else 400, out)

        def log_message(self, *_args):  # quiet
            pass

    server = HTTPServer((host, port), Handler)
    print(f"Graph tool server on http://{host}:{port}  (GET /tools, POST /tools/call)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
