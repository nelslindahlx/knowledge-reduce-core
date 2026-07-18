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


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>KnowledgeReduce - Interactive Visual Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Mono&display=swap" rel="stylesheet">
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: rgba(17, 24, 39, 0.7);
            --border-glow: rgba(147, 51, 234, 0.3);
            --accent-glow: #9333ea;
            --accent-cyan: #06b6d4;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        body {
            margin: 0;
            padding: 0;
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            overflow: hidden;
            display: flex;
            height: 100vh;
        }

        /* Sidebar glassmorphic panel */
        .sidebar {
            width: 380px;
            background: var(--bg-secondary);
            backdrop-filter: blur(16px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            display: flex;
            flex-direction: column;
            padding: 24px;
            box-sizing: border-box;
            z-index: 10;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }

        .logo-area h1 {
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-glow));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.5px;
        }

        .logo-badge {
            background: rgba(147, 51, 234, 0.2);
            border: 1px solid var(--accent-glow);
            border-radius: 6px;
            color: var(--accent-glow);
            padding: 2px 6px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .search-container {
            margin-bottom: 20px;
            position: relative;
        }

        .search-input {
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 10px 14px;
            box-sizing: border-box;
            color: var(--text-main);
            font-family: inherit;
            font-size: 14px;
            outline: none;
            transition: all 0.3s ease;
        }

        .search-input:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.25);
        }

        .panel-title {
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin: 16px 0 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 4px;
        }

        /* Detail display card */
        .detail-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 16px;
            flex-grow: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 20px;
        }

        .detail-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .detail-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }

        .detail-value {
            font-size: 14px;
            font-weight: 400;
            line-height: 1.4;
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .badge-verified { background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #10b981; }
        .badge-likely { background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; color: #3b82f6; }
        .badge-possibly { background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color: #f59e0b; }
        .badge-unverified { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #ef4444; }

        /* Cypher box */
        .cypher-area {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .cypher-input {
            width: 100%;
            height: 60px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            padding: 8px;
            color: var(--text-main);
            font-family: 'Space Mono', monospace;
            font-size: 12px;
            resize: none;
            box-sizing: border-box;
        }

        .cypher-btn {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-glow));
            border: none;
            border-radius: 6px;
            color: white;
            padding: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .cypher-btn:hover {
            opacity: 0.9;
        }

        /* Graph viewport */
        .viewport-container {
            flex-grow: 1;
            position: relative;
            height: 100vh;
        }

        #mynetwork {
            width: 100%;
            height: 100%;
            background-color: var(--bg-primary);
        }

        /* Overlay loader / stats indicator */
        .overlay-stats {
            position: absolute;
            top: 24px;
            right: 24px;
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 14px 20px;
            display: flex;
            gap: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            z-index: 5;
        }

        .stat-box {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .stat-num {
            font-size: 20px;
            font-weight: 800;
            color: var(--accent-cyan);
        }

        .stat-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 2px;
        }

        /* Cypher result modal/panel at bottom */
        .result-panel {
            position: absolute;
            bottom: 24px;
            right: 24px;
            left: 404px;
            max-height: 250px;
            background: rgba(17, 24, 39, 0.9);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            display: none;
            flex-direction: column;
            z-index: 5;
            overflow: hidden;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .result-title {
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--accent-cyan);
        }

        .close-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 18px;
        }

        .table-container {
            overflow: auto;
            flex-grow: 1;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            font-family: 'Space Mono', monospace;
            text-align: left;
        }

        th {
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
        }

        td {
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: var(--text-main);
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 3px;
        }
    </style>
</head>
<body>

    <div class="sidebar">
        <div class="logo-area">
            <h1>KnowledgeReduce</h1>
            <div class="logo-badge">Graph</div>
        </div>

        <div class="search-container">
            <input type="text" class="search-input" id="search-box" placeholder="Search facts, subjects, domains...">
        </div>

        <div class="panel-title">Selected Fact Details</div>
        <div class="detail-card" id="detail-card">
            <div style="text-align: center; color: var(--text-muted); margin-top: 40%; font-size: 13px;">
                Click any node or relationship in the network to inspect the distilled fact details.
            </div>
        </div>

        <div class="panel-title">Run Cypher Query</div>
        <div class="cypher-area">
            <textarea class="cypher-input" id="cypher-box" placeholder="MATCH (f:Fact) RETURN f.statement LIMIT 5"></textarea>
            <button class="cypher-btn" id="cypher-run-btn">Execute Cypher</button>
        </div>
    </div>

    <div class="viewport-container">
        <div class="overlay-stats">
            <div class="stat-box">
                <span class="stat-num" id="total-facts-stat">-</span>
                <span class="stat-label">Total Facts</span>
            </div>
            <div class="stat-box">
                <span class="stat-num" id="total-domains-stat">-</span>
                <span class="stat-label">Domains</span>
            </div>
            <div class="stat-box">
                <span class="stat-num" id="avg-agreement-stat">-</span>
                <span class="stat-label">Avg Agreement</span>
            </div>
        </div>

        <div id="mynetwork"></div>

        <div class="result-panel" id="result-panel">
            <div class="result-header">
                <span class="result-title" id="result-title">Cypher Execution Results</span>
                <button class="close-btn" onclick="document.getElementById('result-panel').style.display = 'none'">&times;</button>
            </div>
            <div class="table-container" id="table-container">
                <!-- Dynamic result table -->
            </div>
        </div>
    </div>

    <script type="text/javascript">
        let network = null;
        let allNodes = [];
        let allEdges = [];
        let nodesDataset = new vis.DataSet();
        let edgesDataset = new vis.DataSet();

        // Color coding for reliability
        const reliabilityColors = {
            'VERIFIED': '#10b981',
            'LIKELY_TRUE': '#3b82f6',
            'POSSIBLY_TRUE': '#f59e0b',
            'UNVERIFIED': '#ef4444'
        };

        // Fetch graph data from API
        async function loadGraph() {
            try {
                const response = await fetch('/api/graph');
                const data = await response.json();
                
                allNodes = data.nodes || [];
                allEdges = data.edges || [];
                
                renderNetwork(allNodes, allEdges);
                updateStats(allNodes);
            } catch (err) {
                console.error("Failed to load graph data:", err);
            }
        }

        function updateStats(nodes) {
            document.getElementById('total-facts-stat').innerText = nodes.length;
            
            const domains = new Set(nodes.map(n => n.domain).filter(Boolean));
            document.getElementById('total-domains-stat').innerText = domains.size;
            
            const agreements = nodes.map(n => n.agreement || 1);
            const avg = agreements.length ? (agreements.reduce((a,b) => a+b, 0) / agreements.length).toFixed(1) : 1;
            document.getElementById('avg-agreement-stat').innerText = avg;
        }

        function renderNetwork(nodes, edges) {
            const visNodes = [];
            const visEdges = [];
            const subjectsAndObjects = new Map();

            nodes.forEach(f => {
                if (f.subject) {
                    const sLower = f.subject.trim().toLowerCase();
                    if (!subjectsAndObjects.has(sLower)) {
                        subjectsAndObjects.set(sLower, { id: 'concept_' + sLower, label: f.subject.trim(), type: 'concept' });
                    }
                }
                if (f.object) {
                    const oLower = f.object.trim().toLowerCase();
                    if (!subjectsAndObjects.has(oLower)) {
                        subjectsAndObjects.set(oLower, { id: 'concept_' + oLower, label: f.object.trim(), type: 'concept' });
                    }
                }
            });

            subjectsAndObjects.forEach(concept => {
                visNodes.push({
                    id: concept.id,
                    label: concept.label,
                    shape: 'dot',
                    size: 15,
                    font: { color: '#f3f4f6', size: 14, face: 'Outfit' },
                    color: {
                        background: '#1f2937',
                        border: '#06b6d4',
                        highlight: { background: '#06b6d4', border: '#06b6d4' }
                    },
                    borderWidth: 2
                });
            });

            nodes.forEach((f, idx) => {
                if (f.subject && f.object) {
                    const fromId = 'concept_' + f.subject.trim().toLowerCase();
                    const toId = 'concept_' + f.object.trim().toLowerCase();
                    const relColor = reliabilityColors[f.reliability] || '#9ca3af';
                    
                    visEdges.push({
                        id: 'fact_' + f.block_id,
                        from: fromId,
                        to: toId,
                        label: f.predicate,
                        color: { color: relColor, highlight: '#9333ea' },
                        font: { align: 'middle', size: 10, face: 'Outfit', color: '#9ca3af' },
                        arrows: 'to',
                        width: 2,
                        smooth: { type: 'curvedCW', roundness: 0.15 }
                    });
                }
            });

            nodesDataset.clear();
            edgesDataset.clear();
            nodesDataset.add(visNodes);
            edgesDataset.add(visEdges);

            const container = document.getElementById('mynetwork');
            const data = { nodes: nodesDataset, edges: edgesDataset };
            
            const options = {
                physics: {
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {
                        gravitationalConstant: -50,
                        centralGravity: 0.01,
                        springLength: 100,
                        springConstant: 0.08
                    },
                    stabilization: { iterations: 150 }
                },
                interaction: { hover: true, selectConnectedEdges: true }
            };
            
            network = new vis.Network(container, data, options);

            network.on("selectEdge", function(params) {
                if (params.edges.length === 1) {
                    const edgeId = params.edges[0];
                    if (edgeId.startsWith('fact_')) {
                        const blockId = edgeId.replace('fact_', '');
                        const fact = nodes.find(n => n.block_id === blockId);
                        if (fact) {
                            showFactDetails(fact);
                        }
                    }
                }
            });

            network.on("selectNode", function(params) {
                if (params.nodes.length === 1) {
                    const nodeId = params.nodes[0];
                    const conceptName = nodeId.replace('concept_', '');
                    const relatedFacts = nodes.filter(n => 
                        (n.subject && n.subject.trim().toLowerCase() === conceptName) ||
                        (n.object && n.object.trim().toLowerCase() === conceptName)
                    );
                    showConceptDetails(conceptName, relatedFacts);
                }
            });
        }

        function showFactDetails(fact) {
            const card = document.getElementById('detail-card');
            const badgeClass = 'badge badge-' + fact.reliability.toLowerCase().replace('_true', '');
            const modelsList = fact.source_models ? fact.source_models.split(',').map(m => `<code>${m.trim()}</code>`).join(', ') : 'None';
            
            card.innerHTML = `
                <div class="detail-item">
                    <span class="detail-label">Distilled Fact</span>
                    <span class="detail-value" style="font-size:16px; font-weight:600; color:var(--accent-cyan);">${fact.statement}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Subject</span>
                    <span class="detail-value"><code>${fact.subject}</code></span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Predicate</span>
                    <span class="detail-value"><code>${fact.predicate}</code></span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Object</span>
                    <span class="detail-value"><code>${fact.object}</code></span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Reliability Tier</span>
                    <div><span class="${badgeClass}">${fact.reliability}</span></div>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Quality Metrics</span>
                    <span class="detail-value">Agreement: <b>${fact.agreement}</b> | Quality Score: <b>${fact.quality}</b></span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Domain</span>
                    <span class="detail-value">${fact.domain || 'N/A'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Source Models</span>
                    <span class="detail-value">${modelsList}</span>
                </div>
            `;
        }

        function showConceptDetails(concept, facts) {
            const card = document.getElementById('detail-card');
            let factsListHTML = facts.map(f => 
                `<div style="font-size:12px; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.03); padding-bottom:6px;">
                    <span style="color:var(--accent-cyan)">${f.statement}</span>
                    <div style="font-size:10px; color:var(--text-muted); margin-top:2px;">Reliability: ${f.reliability}</div>
                </div>`
            ).join('');

            card.innerHTML = `
                <div class="detail-item">
                    <span class="detail-label">Concept Entity</span>
                    <span class="detail-value" style="font-size:18px; font-weight:600; text-transform:capitalize;">${concept}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Involved Facts (${facts.length})</span>
                    <div style="margin-top:8px; max-height:220px; overflow-y:auto;">
                        ${factsListHTML || 'No direct facts.'}
                    </div>
                </div>
            `;
        }

        document.getElementById('search-box').addEventListener('input', function(e) {
            const term = e.target.value.toLowerCase().trim();
            if (!term) {
                renderNetwork(allNodes, allEdges);
                return;
            }
            
            const filteredNodes = allNodes.filter(n => 
                (n.subject && n.subject.toLowerCase().includes(term)) ||
                (n.object && n.object.toLowerCase().includes(term)) ||
                (n.predicate && n.predicate.toLowerCase().includes(term)) ||
                (n.statement && n.statement.toLowerCase().includes(term)) ||
                (n.domain && n.domain.toLowerCase().includes(term))
            );
            
            renderNetwork(filteredNodes, allEdges);
        });

        document.getElementById('cypher-run-btn').addEventListener('click', async function() {
            const cypher = document.getElementById('cypher-box').value.trim();
            if (!cypher) return;
            
            try {
                const response = await fetch('/tools/call', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: 'graph_query',
                        arguments: { cypher: cypher, limit: 100 }
                    })
                });
                const data = await response.json();
                
                const panel = document.getElementById('result-panel');
                const container = document.getElementById('table-container');
                panel.style.display = 'flex';
                
                if (!data.ok) {
                    container.innerHTML = `<div style="color:#ef4444; font-family:monospace; font-size:12px;">Error: ${data.error}</div>`;
                    return;
                }
                
                const rows = data.result || [];
                if (rows.length === 0) {
                    container.innerHTML = `<div style="color:var(--text-muted); font-size:12px;">Query succeeded but returned 0 rows.</div>`;
                    return;
                }
                
                const cols = Object.keys(rows[0]);
                let tableHTML = '<table><thead><tr>';
                cols.forEach(c => tableHTML += `<th>${c}</th>`);
                tableHTML += '</tr></thead><tbody>';
                
                rows.forEach(r => {
                    tableHTML += '<tr>';
                    cols.forEach(c => {
                        let val = r[c];
                        if (typeof val === 'object' && val !== null) {
                            val = JSON.stringify(val);
                        }
                        tableHTML += `<td>${val !== null ? val : ''}</td>`;
                    });
                    tableHTML += '</tr>';
                });
                tableHTML += '</tbody></table>';
                
                container.innerHTML = tableHTML;
            } catch (err) {
                console.error("Cypher error:", err);
            }
        });

        loadGraph();
    </script>
</body>
</html>"""


def make_handler(tools: Any) -> Any:
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path_clean = self.path.split("?")[0].rstrip("/")
            if path_clean == "/tools":
                self._send(200, {"tools": list_tools()})
            elif path_clean in ("", "/dashboard"):
                self._send_html(DASHBOARD_HTML)
            elif path_clean == "/api/graph":
                try:
                    nodes = tools.store.query(
                        "MATCH (f:Fact) "
                        "RETURN f.block_id AS block_id, f.statement AS statement, f.subject AS subject, "
                        "f.predicate AS predicate, f.object AS object, f.domain AS domain, "
                        "f.reliability AS reliability, f.agreement AS agreement, f.quality AS quality, "
                        "f.source_models AS source_models"
                    )
                    edges = tools.store.query(
                        "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
                        "RETURN a.block_id AS from_id, b.block_id AS to_id, r.predicate AS predicate, r.weight AS weight"
                    )
                    self._send(200, {"nodes": nodes, "edges": edges})
                except Exception as exc:
                    self._send(500, {"error": str(exc)})
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

    return Handler


def serve(store_path: str, host: str = "127.0.0.1", port: int = 8080) -> None:  # pragma: no cover
    """Start the HTTP tool server backed by a KuzuStore at ``store_path``.

    Blocking call -- runs until interrupted. Imported lazily so the module
    stays importable (and testable) without kuzu installed.
    """
    from http.server import HTTPServer
    from .kuzu_store import KuzuStore
    from .graph_tool import GraphTools

    tools = GraphTools(KuzuStore(store_path))
    Handler = make_handler(tools)
    server = HTTPServer((host, port), Handler)
    print(f"Graph tool server on http://{host}:{port}  (GET /tools, POST /tools/call)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
