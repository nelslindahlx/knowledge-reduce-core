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

from collections import OrderedDict
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from .graph_tool import TOOL_SCHEMAS

# Required arguments per tool (mirrors TOOL_SCHEMAS "required").
_REQUIRED = {t["name"]: t["parameters"].get("required", []) for t in TOOL_SCHEMAS}
_WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_BODY_BYTES = 1_048_576
_MAX_WORKSPACE_CACHE = 16


def _validate_workspace_id(workspace_id: str) -> str:
    cleaned = (workspace_id or "default").strip() or "default"
    if cleaned == "default":
        return "default"
    if not _WORKSPACE_ID_RE.fullmatch(cleaned):
        raise ValueError("Invalid workspace ID; must match [A-Za-z0-9_-]{1,64}.")
    return cleaned


def _verify_jwt_token(token: Optional[str]) -> Optional[str]:
    """Verify Bearer JWT token when MCP_JWT_SECRET is configured.

    Returns ``None`` on success (or when auth is disabled), or an error message.
    """
    secret = os.environ.get("MCP_JWT_SECRET")
    if not secret:
        return None
    if not token:
        return "missing bearer token"
    try:
        import jwt

        jwt.decode(
            token,
            secret,
            algorithms=["HS256", "HS384", "HS512"],
            options={"require": ["exp"]},
        )
        return None
    except Exception:
        return "invalid token signature or expired claim"


def list_tools() -> list:
    """Return the tool schemas (for LLM function-calling / MCP registration)."""
    return TOOL_SCHEMAS


def handle_tool_call(tools: Any, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch one tool call against a GraphTools-like object.

    Returns ``{"ok": True, "result": [...]}`` or ``{"ok": False, "error": ...}``.
    Never raises -- errors are returned as data so the server can always reply.
    """
    if not isinstance(name, str):
        return {"ok": False, "error": "tool name must be a string"}
    method = getattr(tools, name, None)
    if method is None or name not in _REQUIRED:
        return {"ok": False, "error": f"unknown tool: {name}"}
    if arguments is None:
        arguments = {}
    elif not isinstance(arguments, dict):
        return {"ok": False, "error": "arguments must be a JSON object"}
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
    <script src="https://unpkg.com/3d-force-graph@1.73.4/dist/3d-force-graph.min.js" crossorigin="anonymous"></script>
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
            margin-bottom: 16px;
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
            margin-bottom: 14px;
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
            margin: 12px 0 8px 0;
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
            margin-bottom: 16px;
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

        .prune-btn {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            color: #ef4444;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
            margin-top: 10px;
        }
        .prune-btn:hover {
            background: #ef4444;
            color: #fff;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
        }

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
        .node-tooltip, .link-tooltip {
            background: rgba(17, 24, 39, 0.95);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 8px 12px;
            border-radius: 8px;
            color: #f3f4f6;
            font-size: 12px;
            font-family: 'Outfit', sans-serif;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            pointer-events: none;
        }
        .node-tooltip b, .link-tooltip b {
            color: var(--accent-cyan);
        }
    </style>
</head>
<body>

    <div class="sidebar">
        <div class="logo-area">
            <h1>KnowledgeReduce</h1>
            <div class="logo-badge">Graph</div>
        </div>

        <div style="display: flex; gap: 8px; margin-bottom: 12px;">
            <button class="cypher-btn" id="mode-instance-btn" style="flex: 1; background: #3b82f6; border: none; padding: 6px 12px; font-size: 11px; font-weight: 600; cursor: pointer; border-radius: 4px;">Instance Graph</button>
            <button class="cypher-btn" id="mode-schema-btn" style="flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 6px 12px; font-size: 11px; font-weight: 600; color: var(--text-muted); cursor: pointer; border-radius: 4px;">Ontology Schema</button>
        </div>

        <div class="search-container">
            <input type="text" class="search-input" id="search-box" placeholder="Search facts, subjects, domains...">
        </div>

        <div class="search-container" style="display: flex; gap: 6px;">
            <input type="password" class="search-input" id="auth-token-box" placeholder="Bearer JWT (optional)" style="font-size: 12px; padding: 6px 10px;">
            <button class="cypher-btn" id="auth-apply-btn" style="padding: 6px 10px; font-size: 11px;">Auth</button>
        </div>

        <div class="panel-title">Graph-RAG Retriever</div>
        <div style="padding: 15px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.05);">
            <input type="text" class="search-input" id="rag-query-box" placeholder="Ask the knowledge graph..." style="margin-bottom: 10px; width: 100%;">
            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                <label style="font-size: 11px; color: var(--text-muted);">
                    Hops: <select id="rag-hops" style="background: #1f2937; color: white; border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; padding: 2px;">
                        <option value="1">1</option>
                        <option value="2" selected>2</option>
                        <option value="3">3</option>
                    </select>
                </label>
                <label style="font-size: 11px; color: var(--text-muted); display: flex; align-items: center; gap: 4px;">
                    <input type="checkbox" id="rag-exclude-unverified" checked style="accent-color: #3b82f6;"> Exclude Unverified
                </label>
            </div>
            <button class="cypher-btn" id="rag-run-btn" style="width: 100%; background: linear-gradient(135deg, #8b5cf6, #3b82f6); border: none;">Retrieve RAG Context</button>
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
                <button class="close-btn" id="result-close-btn">&times;</button>
            </div>
            <div class="table-container" id="table-container">
                <!-- Dynamic result table -->
            </div>
        </div>
    </div>

    <script type="text/javascript">
        let Graph = null;
        let allNodes = [];
        let allEdges = [];
        let activeFacts = [];
        let highlightedNodes = new Set();
        let highlightedLinks = new Set();
        let currentMode = 'instance';
        let activeWs = null;

        const reliabilityColors = {
            'VERIFIED': '#10b981',
            'LIKELY_TRUE': '#3b82f6',
            'POSSIBLY_TRUE': '#f59e0b',
            'UNVERIFIED': '#ef4444'
        };

        function escapeHtml(text) {
            if (text === null || text === undefined) {
                return '';
            }
            return String(text)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function getAuthToken() {
            const box = document.getElementById('auth-token-box');
            const boxToken = box ? box.value.trim() : '';
            if (boxToken) return boxToken;
            return (sessionStorage.getItem('kr_token') || '').trim();
        }

        function authHeaders(extra) {
            const headers = Object.assign({}, extra || {});
            const tok = getAuthToken();
            if (tok) {
                headers['Authorization'] = `Bearer ${tok}`;
            }
            return headers;
        }

        function setModeButtons(mode) {
            currentMode = mode;
            const instBtn = document.getElementById('mode-instance-btn');
            const schBtn = document.getElementById('mode-schema-btn');
            if (mode === 'instance') {
                instBtn.style.background = '#3b82f6';
                instBtn.style.color = '#ffffff';
                instBtn.style.border = 'none';
                schBtn.style.background = 'rgba(255,255,255,0.05)';
                schBtn.style.color = 'var(--text-muted)';
                schBtn.style.border = '1px solid rgba(255,255,255,0.1)';
            } else {
                schBtn.style.background = '#3b82f6';
                schBtn.style.color = '#ffffff';
                schBtn.style.border = 'none';
                instBtn.style.background = 'rgba(255,255,255,0.05)';
                instBtn.style.color = 'var(--text-muted)';
                instBtn.style.border = '1px solid rgba(255,255,255,0.1)';
            }
        }

        function ensureGraphInstance() {
            const container = document.getElementById('mynetwork');
            if (!Graph) {
                container.innerHTML = '';
                Graph = ForceGraph3D()(container)
                    .backgroundColor('#0b0f19')
                    .linkDirectionalArrowRelPos(0.95)
                    .linkCurvature(0.15);
            }
            return Graph;
        }

        async function loadGraph() {
            try {
                const response = await fetch('/api/graph', { headers: authHeaders() });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
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

            const agreements = nodes.map(n => Number(n.agreement) || 1);
            const avg = agreements.length ? (agreements.reduce((a, b) => a + b, 0) / agreements.length).toFixed(1) : '1.0';
            document.getElementById('avg-agreement-stat').innerText = avg;
        }

        function renderNetwork(nodes, edges) {
            activeFacts = nodes;
            const visNodes = [];
            const visLinks = [];
            const subjectsAndObjects = new Map();

            nodes.forEach(f => {
                if (f.subject) {
                    const sTrim = String(f.subject).trim();
                    const sLower = sTrim.toLowerCase();
                    if (!subjectsAndObjects.has(sLower)) {
                        subjectsAndObjects.set(sLower, {
                            id: 'concept_' + sLower,
                            label: sTrim,
                            type: 'concept',
                            semType: f.subject_type || 'CONCEPT',
                            parentClass: f.subject_parent || ''
                        });
                    }
                }
                if (f.object) {
                    const oTrim = String(f.object).trim();
                    const oLower = oTrim.toLowerCase();
                    if (!subjectsAndObjects.has(oLower)) {
                        subjectsAndObjects.set(oLower, {
                            id: 'concept_' + oLower,
                            label: oTrim,
                            type: 'concept',
                            semType: f.object_type || 'CONCEPT',
                            parentClass: f.object_parent || ''
                        });
                    }
                }
            });

            const typeColors = {
                'PROCESS': '#10b981',
                'ENTITY': '#3b82f6',
                'LOCATION': '#f59e0b',
                'ATTRIBUTE': '#ec4899',
                'CONCEPT': '#06b6d4'
            };

            subjectsAndObjects.forEach(concept => {
                const color = typeColors[concept.semType] || '#06b6d4';
                const labelText = concept.parentClass ? `${concept.label} (${concept.parentClass})` : concept.label;
                visNodes.push({
                    id: concept.id,
                    label: labelText,
                    color: color,
                    val: 15
                });
            });

            nodes.forEach(f => {
                if (f.subject && f.object) {
                    const fromId = 'concept_' + String(f.subject).trim().toLowerCase();
                    const toId = 'concept_' + String(f.object).trim().toLowerCase();
                    const rel = f.reliability || 'UNVERIFIED';
                    const relColor = reliabilityColors[rel] || '#9ca3af';

                    visLinks.push({
                        id: 'fact_' + f.block_id,
                        source: fromId,
                        target: toId,
                        label: `${f.predicate || ''} (${rel})`,
                        color: relColor,
                        fact: f
                    });
                }
            });

            const g = ensureGraphInstance();
            g.nodeColor(node => {
                    if (highlightedNodes.size > 0) {
                        return highlightedNodes.has(node.id) ? '#fbbf24' : '#1f2937';
                    }
                    return node.color;
                })
                .nodeLabel(node => `<div class="node-tooltip"><b>${escapeHtml(node.label)}</b></div>`)
                .nodeVal(node => {
                    if (highlightedNodes.size > 0) {
                        return highlightedNodes.has(node.id) ? 25 : 8;
                    }
                    return node.val;
                })
                .linkColor(link => {
                    if (highlightedLinks.size > 0) {
                        return highlightedLinks.has(link.id) ? '#fbbf24' : '#1f2937';
                    }
                    return link.color;
                })
                .linkWidth(link => {
                    if (highlightedLinks.size > 0) {
                        return highlightedLinks.has(link.id) ? 4 : 1.5;
                    }
                    return 1.5;
                })
                .linkLabel(link => `<div class="link-tooltip"><b>${escapeHtml(link.label)}</b><br/>${escapeHtml(link.fact ? link.fact.statement : '')}</div>`)
                .linkDirectionalArrowLength(4)
                .onNodeClick(node => {
                    const distance = 80;
                    const mag = Math.hypot(node.x || 0, node.y || 0, node.z || 0);
                    if (mag > 0) {
                        const distRatio = 1 + distance / mag;
                        g.cameraPosition(
                            { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
                            node,
                            1500
                        );
                    }

                    const conceptName = String(node.id || '').replace('concept_', '');
                    const relatedFacts = activeFacts.filter(n =>
                        (n.subject && String(n.subject).trim().toLowerCase() === conceptName) ||
                        (n.object && String(n.object).trim().toLowerCase() === conceptName)
                    );
                    showConceptDetails(conceptName, relatedFacts);
                })
                .onLinkClick(link => {
                    if (link && link.fact) {
                        showFactDetails(link.fact);
                    }
                })
                .graphData({ nodes: visNodes, links: visLinks });
        }

        function showFactDetails(fact) {
            const card = document.getElementById('detail-card');
            const rel = String(fact.reliability || 'UNVERIFIED');
            const badgeSuffix = rel.toLowerCase().replace('_true', '');
            const allowedSuffixes = new Set(['verified', 'likely', 'possibly', 'unverified']);
            const safeSuffix = allowedSuffixes.has(badgeSuffix) ? badgeSuffix : 'unverified';
            const badgeClass = 'badge badge-' + safeSuffix;

            const rawModels = Array.isArray(fact.source_models)
                ? fact.source_models
                : (fact.source_models ? String(fact.source_models).split(',') : []);
            const cleanedModels = rawModels.map(m => String(m).trim()).filter(Boolean);
            const modelsList = cleanedModels.length
                ? cleanedModels.map(m => `<code>${escapeHtml(m)}</code>`).join(', ')
                : 'None';

            card.innerHTML = `
                <div class="detail-item">
                    <span class="detail-label">Distilled Fact</span>
                    <span class="detail-value" style="font-size:16px; font-weight:600; color:var(--accent-cyan);">${escapeHtml(fact.statement)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Subject</span>
                    <span class="detail-value"><code>${escapeHtml(fact.subject)}</code></span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Predicate</span>
                    <span class="detail-value"><code>${escapeHtml(fact.predicate)}</code></span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Object</span>
                    <span class="detail-value"><code>${escapeHtml(fact.object)}</code></span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Reliability Tier</span>
                    <div><span class="${badgeClass}">${escapeHtml(rel)}</span></div>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Quality Metrics</span>
                    <span class="detail-value">Agreement: <b>${escapeHtml(fact.agreement ?? 1)}</b> | Quality Score: <b>${escapeHtml(fact.quality ?? 0)}</b></span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Domain</span>
                    <span class="detail-value">${escapeHtml(fact.domain || 'N/A')}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Source Models</span>
                    <span class="detail-value">${modelsList}</span>
                </div>
                <div class="detail-item" style="margin-top:16px;">
                    <button class="prune-btn" data-block-id="${escapeHtml(fact.block_id)}">Prune Fact</button>
                </div>
            `;
            const pruneBtn = card.querySelector('.prune-btn');
            if (pruneBtn) {
                pruneBtn.addEventListener('click', () => pruneFact(fact.block_id));
            }
        }

        function showConceptDetails(concept, facts) {
            const card = document.getElementById('detail-card');
            const factsListHTML = facts.map(f =>
                `<div style="font-size:12px; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.03); padding-bottom:6px;">
                    <span style="color:var(--accent-cyan)">${escapeHtml(f.statement)}</span>
                    <div style="font-size:10px; color:var(--text-muted); margin-top:2px;">Reliability: ${escapeHtml(f.reliability || 'UNVERIFIED')}</div>
                </div>`
            ).join('');

            card.innerHTML = `
                <div class="detail-item">
                    <span class="detail-label">Concept Entity</span>
                    <span class="detail-value" style="font-size:18px; font-weight:600; text-transform:capitalize;">${escapeHtml(concept)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Involved Facts (${facts.length})</span>
                    <div style="margin-top:8px; max-height:220px; overflow-y:auto;">
                        ${factsListHTML || 'No direct facts.'}
                    </div>
                </div>
            `;
        }

        async function pruneFact(blockId) {
            if (!blockId || !confirm("Are you sure you want to prune this fact from the knowledge graph?")) {
                return;
            }
            try {
                const response = await fetch('/api/facts/' + encodeURIComponent(blockId), {
                    method: 'DELETE',
                    headers: authHeaders()
                });
                const resData = await response.json();
                if (response.ok && resData.ok) {
                    allNodes = allNodes.filter(n => n.block_id !== blockId);
                    allEdges = allEdges.filter(e => e.from_id !== blockId && e.to_id !== blockId);
                    if (currentMode === 'instance') {
                        renderNetwork(allNodes, allEdges);
                    }
                    updateStats(allNodes);
                    document.getElementById('detail-card').innerHTML = '<div style="color:var(--text-muted); font-size:14px; font-style:italic; text-align:center; margin-top:40px;">Fact pruned successfully. Select another fact or node to view details.</div>';
                } else {
                    alert("Error pruning fact: " + (resData.error || response.statusText));
                }
            } catch (err) {
                console.error("Pruning failed:", err);
                alert("Pruning failed: " + err);
            }
        }

        document.getElementById('auth-apply-btn').addEventListener('click', () => {
            const box = document.getElementById('auth-token-box');
            const tok = box ? box.value.trim() : '';
            if (tok) {
                sessionStorage.setItem('kr_token', tok);
            } else {
                sessionStorage.removeItem('kr_token');
            }
            loadGraph();
            setupWebSocket();
        });

        document.getElementById('result-close-btn').addEventListener('click', () => {
            document.getElementById('result-panel').style.display = 'none';
        });

        document.getElementById('search-box').addEventListener('input', function(e) {
            const term = e.target.value.toLowerCase().trim();
            if (currentMode !== 'instance') {
                setModeButtons('instance');
            }
            if (!term) {
                renderNetwork(allNodes, allEdges);
                return;
            }

            const filteredNodes = allNodes.filter(n =>
                (n.subject && String(n.subject).toLowerCase().includes(term)) ||
                (n.object && String(n.object).toLowerCase().includes(term)) ||
                (n.predicate && String(n.predicate).toLowerCase().includes(term)) ||
                (n.statement && String(n.statement).toLowerCase().includes(term)) ||
                (n.domain && String(n.domain).toLowerCase().includes(term))
            );

            renderNetwork(filteredNodes, allEdges);
        });

        document.getElementById('cypher-run-btn').addEventListener('click', async function() {
            const cypher = document.getElementById('cypher-box').value.trim();
            if (!cypher) return;

            const panel = document.getElementById('result-panel');
            const container = document.getElementById('table-container');
            document.getElementById('result-title').innerText = 'Cypher Execution Results';

            try {
                const response = await fetch('/tools/call', {
                    method: 'POST',
                    headers: authHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({
                        name: 'graph_query',
                        arguments: { cypher: cypher, limit: 100 }
                    })
                });
                const data = await response.json();
                panel.style.display = 'flex';

                if (!response.ok || !data.ok) {
                    container.innerHTML = `<div style="color:#ef4444; font-family:monospace; font-size:12px;">Error: ${escapeHtml(data.error || response.statusText)}</div>`;
                    return;
                }

                const rows = data.result || [];
                if (rows.length === 0) {
                    container.innerHTML = `<div style="color:var(--text-muted); font-size:12px;">Query succeeded but returned 0 rows.</div>`;
                    return;
                }

                const cols = Object.keys(rows[0]);
                let tableHTML = '<table><thead><tr>';
                cols.forEach(c => { tableHTML += `<th>${escapeHtml(c)}</th>`; });
                tableHTML += '</tr></thead><tbody>';

                rows.forEach(r => {
                    tableHTML += '<tr>';
                    cols.forEach(c => {
                        let val = r[c];
                        if (typeof val === 'object' && val !== null) {
                            val = JSON.stringify(val);
                        }
                        tableHTML += `<td>${escapeHtml(val !== null && val !== undefined ? val : '')}</td>`;
                    });
                    tableHTML += '</tr>';
                });
                tableHTML += '</tbody></table>';

                container.innerHTML = tableHTML;
            } catch (err) {
                panel.style.display = 'flex';
                container.innerHTML = `<div style="color:#ef4444; font-family:monospace; font-size:12px;">Error: ${escapeHtml(err.message || err)}</div>`;
            }
        });

        document.getElementById('rag-run-btn').addEventListener('click', async () => {
            const query = document.getElementById('rag-query-box').value.trim();
            if (!query) {
                highlightedNodes.clear();
                highlightedLinks.clear();
                if (Graph) {
                    Graph.nodeColor(Graph.nodeColor())
                        .linkColor(Graph.linkColor())
                        .linkWidth(Graph.linkWidth())
                        .nodeVal(Graph.nodeVal());
                }
                return;
            }

            const hops = document.getElementById('rag-hops').value;
            const excludeUnverified = document.getElementById('rag-exclude-unverified').checked;

            try {
                const response = await fetch(
                    `/api/rag_retrieve?query=${encodeURIComponent(query)}&hops=${encodeURIComponent(hops)}&exclude_unverified=${encodeURIComponent(excludeUnverified)}`,
                    { headers: authHeaders() }
                );
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();

                if (currentMode !== 'instance') {
                    setModeButtons('instance');
                    renderNetwork(allNodes, allEdges);
                }

                highlightedNodes.clear();
                highlightedLinks.clear();

                (data.facts || []).forEach(f => {
                    if (f.subject) highlightedNodes.add('concept_' + String(f.subject).trim().toLowerCase());
                    if (f.object) highlightedNodes.add('concept_' + String(f.object).trim().toLowerCase());
                    highlightedLinks.add('fact_' + f.block_id);
                });

                if (Graph) {
                    Graph.nodeColor(Graph.nodeColor())
                        .linkColor(Graph.linkColor())
                        .linkWidth(Graph.linkWidth())
                        .nodeVal(Graph.nodeVal());
                }

                const resultPanel = document.getElementById('result-panel');
                document.getElementById('result-title').innerText = `Graph-RAG Context: "${query}"`;
                const container = document.getElementById('table-container');
                container.innerHTML = `
                    <div style="padding: 15px; color: #f3f4f6; font-family: monospace; white-space: pre-wrap; font-size: 12px; background: rgba(0,0,0,0.2); border-radius: 6px; line-height: 1.5;">${escapeHtml(data.context)}</div>
                `;
                resultPanel.style.display = 'flex';
            } catch (err) {
                alert('Error running Graph-RAG: ' + err.message);
            }
        });

        document.getElementById('mode-instance-btn').addEventListener('click', () => {
            if (currentMode === 'instance') return;
            setModeButtons('instance');
            renderNetwork(allNodes, allEdges);
        });

        document.getElementById('mode-schema-btn').addEventListener('click', async () => {
            if (currentMode === 'schema') return;
            setModeButtons('schema');
            const ok = await loadSchemaData();
            if (!ok) {
                setModeButtons('instance');
            }
        });

        async function loadSchemaData() {
            try {
                const response = await fetch('/api/schema', { headers: authHeaders() });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();

                const schemaNodes = new Set();
                const schemaLinks = [];

                (data.schema || []).forEach(s => {
                    schemaNodes.add(s.subject_type);
                    schemaNodes.add(s.object_type);
                    schemaLinks.push({
                        id: `schema_${s.subject_type}_${s.predicate}_${s.object_type}`,
                        source: `concept_${String(s.subject_type).toLowerCase()}`,
                        target: `concept_${String(s.object_type).toLowerCase()}`,
                        label: s.predicate,
                        color: '#a78bfa',
                        fact: {
                            statement: `Schema relation: ${s.subject_type} -> ${s.predicate} -> ${s.object_type}`,
                            subject: s.subject_type,
                            predicate: s.predicate,
                            object: s.object_type,
                            reliability: 'VERIFIED'
                        }
                    });
                });

                const typeColors = {
                    'PROCESS': '#10b981',
                    'ENTITY': '#3b82f6',
                    'LOCATION': '#f59e0b',
                    'ATTRIBUTE': '#ec4899',
                    'CONCEPT': '#06b6d4'
                };

                const visNodes = Array.from(schemaNodes).map(type => ({
                    id: `concept_${String(type).toLowerCase()}`,
                    label: type,
                    color: typeColors[type] || '#06b6d4',
                    val: 30
                }));

                const g = ensureGraphInstance();
                g.nodeColor(node => node.color)
                    .nodeLabel(node => `<div class="node-tooltip"><b>${escapeHtml(node.label)} Category</b></div>`)
                    .nodeVal(node => node.val)
                    .linkColor(link => link.color)
                    .linkWidth(2.5)
                    .linkLabel(link => `<div class="link-tooltip"><b>${escapeHtml(link.label)}</b></div>`)
                    .linkDirectionalArrowLength(6)
                    .onNodeClick(null)
                    .onLinkClick(link => {
                        if (link && link.fact) showFactDetails(link.fact);
                    })
                    .graphData({ nodes: visNodes, links: schemaLinks });
                return true;
            } catch (err) {
                alert('Error loading ontology schema: ' + err.message);
                return false;
            }
        }

        function setupWebSocket() {
            try {
                if (activeWs) {
                    activeWs.onclose = null;
                    activeWs.close();
                    activeWs = null;
                }
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const tok = getAuthToken();
                const queryPart = tok ? `?token=${encodeURIComponent(tok)}` : '';
                const wsUrl = `${protocol}//${window.location.host}/ws${queryPart}`;
                const ws = new WebSocket(wsUrl);
                activeWs = ws;

                ws.onmessage = (event) => {
                    try {
                        const update = JSON.parse(event.data);
                        if (update.type === 'graph_update') {
                            if (update.nodes) {
                                update.nodes.forEach(n => {
                                    if (!allNodes.some(x => x.block_id === n.block_id)) {
                                        allNodes.push(n);
                                    }
                                });
                            }
                            if (update.edges) {
                                update.edges.forEach(e => {
                                    if (!allEdges.some(x => x.from_id === e.from_id && x.to_id === e.to_id)) {
                                        allEdges.push(e);
                                    }
                                });
                            }
                            if (currentMode === 'instance') {
                                renderNetwork(allNodes, allEdges);
                            }
                            updateStats(allNodes);
                        } else if (update.type === 'prune_fact') {
                            const blockId = update.block_id;
                            allNodes = allNodes.filter(n => n.block_id !== blockId);
                            allEdges = allEdges.filter(e => e.from_id !== blockId && e.to_id !== blockId);
                            if (currentMode === 'instance') {
                                renderNetwork(allNodes, allEdges);
                            }
                            updateStats(allNodes);
                        }
                    } catch (e) {
                        console.error("Error processing websocket update:", e);
                    }
                };

                ws.onclose = (ev) => {
                    if (ev.code !== 1008) {
                        setTimeout(setupWebSocket, 5000);
                    }
                };
            } catch (e) {
                console.error("Failed to setup WebSocket:", e);
            }
        }

        loadGraph();
        setupWebSocket();
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

        def _authorize(self) -> bool:
            auth_header = ""
            if getattr(self, "headers", None):
                auth_header = self.headers.get("Authorization") or ""
            token: Optional[str] = None
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:].strip()
            err = _verify_jwt_token(token)
            if err:
                self._send(401, {"ok": False, "error": f"Unauthorized: {err}"})
                return False
            return True

        def do_GET(self):
            path_clean = self.path.split("?")[0].rstrip("/")
            if path_clean in ("", "/dashboard"):
                self._send_html(DASHBOARD_HTML)
                return
            if path_clean == "/tools":
                if not self._authorize():
                    return
                self._send(200, {"tools": list_tools()})
            elif path_clean == "/api/graph":
                if not self._authorize():
                    return
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
            if not self._authorize():
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                self._send(400, {"ok": False, "error": "invalid Content-Length"})
                return
            if length < 0:
                self._send(400, {"ok": False, "error": "invalid Content-Length"})
                return
            if length > _MAX_BODY_BYTES:
                self._send(413, {"ok": False, "error": "payload too large"})
                return
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"ok": False, "error": "invalid JSON"})
                return
            if not isinstance(req, dict):
                self._send(400, {"ok": False, "error": "JSON payload must be an object"})
                return
            out = handle_tool_call(tools, req.get("name", ""), req.get("arguments", {}))
            self._send(200 if out.get("ok") else 400, out)

        def log_message(self, *_args):  # quiet
            pass

    return Handler


def make_fastapi_app(tools: Any) -> Any:
    """Create a FastAPI application serving the visual dashboard and tool endpoints."""
    from fastapi import Depends, FastAPI, HTTPException, Request, Security, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    app = FastAPI(title="KnowledgeReduce API Gateway")

    raw_origins = os.environ.get(
        "MCP_CORS_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080"
    )
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    allow_creds = "*" not in allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    security = HTTPBearer(auto_error=False)
    security_dep = Security(security)
    workspace_tools: OrderedDict[str, Any] = OrderedDict()

    async def verify_token(
        credentials: Optional[HTTPAuthorizationCredentials] = security_dep,
    ):
        token = credentials.credentials if credentials else None
        err = _verify_jwt_token(token)
        if err:
            raise HTTPException(status_code=401, detail=f"Unauthorized: {err}")

    def get_workspace_id(request: Request) -> str:
        raw = request.headers.get("x-workspace-id", "default")
        try:
            return _validate_workspace_id(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_tools_for_workspace(workspace_id: str) -> Any:
        ws_id = _validate_workspace_id(workspace_id)
        if ws_id == "default":
            return tools
        if ws_id in workspace_tools:
            workspace_tools.move_to_end(ws_id)
            return workspace_tools[ws_id]

        from .graph_store_factory import get_graph_store
        from .graph_tool import GraphTools
        from .neo4j_store import Neo4jStore

        if isinstance(tools.store, Neo4jStore):
            ws_store = tools.store.with_database(ws_id)
        else:
            db_path = getattr(tools.store, "path", None) or getattr(tools.store, "db_path", None)
            if not isinstance(db_path, str) or not db_path.strip():
                db_path = "graph_db"
            abs_db = os.path.abspath(db_path)
            base_dir = os.path.dirname(abs_db)
            base_name = os.path.basename(abs_db) or "graph_db"
            ws_path = os.path.abspath(os.path.join(base_dir, f"{base_name}_{ws_id}"))
            if os.path.commonpath([base_dir, ws_path]) != base_dir:
                raise HTTPException(status_code=400, detail="Invalid workspace path")
            ws_store = get_graph_store(ws_path)

        if len(workspace_tools) >= _MAX_WORKSPACE_CACHE:
            _, evicted = workspace_tools.popitem(last=False)
            evicted_store = getattr(evicted, "store", None)
            if evicted_store is not None and hasattr(evicted_store, "close"):
                try:
                    evicted_store.close()
                except Exception:
                    pass

        workspace_tools[ws_id] = GraphTools(ws_store)
        return workspace_tools[ws_id]

    active_websockets: Dict[str, Set[WebSocket]] = {}
    app.state.workspace_tools = workspace_tools

    async def _broadcast_workspace(workspace_id: str, message: Dict[str, Any]) -> int:
        sockets = active_websockets.get(workspace_id, set())
        closed_sockets: Set[WebSocket] = set()
        for ws in list(sockets):
            try:
                await ws.send_json(message)
            except Exception:
                closed_sockets.add(ws)
        for ws in closed_sockets:
            sockets.discard(ws)
        return len(sockets)

    @app.get("/tools", dependencies=[Depends(verify_token)])
    async def get_tools():
        return {"tools": list_tools()}

    @app.post("/tools/call", dependencies=[Depends(verify_token)])
    async def post_tools_call(request: Request):
        try:
            req = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"ok": False, "error": "invalid JSON"})
        if not isinstance(req, dict):
            return JSONResponse(
                status_code=400, content={"ok": False, "error": "JSON payload must be an object"}
            )
        workspace_id = get_workspace_id(request)
        ws_tools = get_tools_for_workspace(workspace_id)
        out = handle_tool_call(ws_tools, req.get("name", ""), req.get("arguments", {}))
        return JSONResponse(status_code=200 if out.get("ok") else 400, content=out)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        return DASHBOARD_HTML

    @app.get("/api/graph", dependencies=[Depends(verify_token)])
    async def get_api_graph(request: Request):
        try:
            workspace_id = get_workspace_id(request)
            ws_tools = get_tools_for_workspace(workspace_id)
            nodes = ws_tools.store.query(
                "MATCH (f:Fact) "
                "RETURN f.block_id AS block_id, f.statement AS statement, f.subject AS subject, "
                "f.predicate AS predicate, f.object AS object, f.domain AS domain, "
                "f.reliability AS reliability, f.agreement AS agreement, f.quality AS quality, "
                "f.source_models AS source_models"
            )
            edges = ws_tools.store.query(
                "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
                "RETURN a.block_id AS from_id, b.block_id AS to_id, r.predicate AS predicate, r.weight AS weight"
            )

            from .ontology import OntologyDistiller

            distiller = OntologyDistiller(ws_tools.store)
            sem_types = distiller.infer_semantic_types(facts=nodes)
            taxonomy = distiller.distill_taxonomy(facts=nodes)

            parent_classes = {}
            for parent, children in taxonomy.items():
                for child in children:
                    parent_classes[child.lower()] = parent

            enriched_nodes = []
            for n in nodes:
                s = str(n.get("subject") or "").strip()
                o = str(n.get("object") or "").strip()
                node_copy = dict(n)
                node_copy["subject_type"] = sem_types.get(s, "CONCEPT")
                node_copy["object_type"] = sem_types.get(o, "CONCEPT")
                node_copy["subject_parent"] = parent_classes.get(s.lower(), None)
                node_copy["object_parent"] = parent_classes.get(o.lower(), None)
                enriched_nodes.append(node_copy)

            return {"nodes": enriched_nodes, "edges": edges}
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/api/rag_retrieve", dependencies=[Depends(verify_token)])
    async def api_rag_retrieve(
        query: str, request: Request, hops: int = 2, exclude_unverified: bool = True
    ):
        try:
            workspace_id = get_workspace_id(request)
            ws_tools = get_tools_for_workspace(workspace_id)
            from .rag import GraphRAGRetriever

            retriever = GraphRAGRetriever(ws_tools.store)
            retrieved_facts = retriever.retrieve(
                query, hops=hops, exclude_unverified=exclude_unverified
            )
            formatted_context = retriever.format_facts(retrieved_facts)
            return {
                "facts": retrieved_facts,
                "context": formatted_context,
            }
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/api/schema", dependencies=[Depends(verify_token)])
    async def get_api_schema(request: Request):
        try:
            workspace_id = get_workspace_id(request)
            ws_tools = get_tools_for_workspace(workspace_id)
            from .ontology import OntologyDistiller

            distiller = OntologyDistiller(ws_tools.store)
            schema = distiller.infer_relation_schema()
            return {"schema": schema}
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        raw_ws = websocket.headers.get("x-workspace-id") or websocket.query_params.get(
            "workspace_id", "default"
        )
        try:
            ws_id = _validate_workspace_id(raw_ws)
        except ValueError:
            await websocket.close(code=1008)
            return

        auth_header = websocket.headers.get("authorization", "")
        token: Optional[str] = None
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if not token:
            token = websocket.query_params.get("token")

        if _verify_jwt_token(token):
            await websocket.close(code=1008)
            return

        await websocket.accept()
        sockets = active_websockets.setdefault(ws_id, set())
        sockets.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            sockets.discard(websocket)

    @app.post("/api/notify_update", dependencies=[Depends(verify_token)])
    async def notify_update(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "invalid JSON"})
        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400, content={"error": "JSON payload must be an object"}
            )

        workspace_id = get_workspace_id(request)
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])

        recipients = await _broadcast_workspace(
            workspace_id,
            {"type": "graph_update", "nodes": nodes, "edges": edges},
        )
        return {"status": "broadcasted", "recipients": recipients}

    @app.delete("/api/facts/{block_id}", dependencies=[Depends(verify_token)])
    async def delete_fact(block_id: str, request: Request):
        try:
            workspace_id = get_workspace_id(request)
            ws_tools = get_tools_for_workspace(workspace_id)
            if hasattr(ws_tools, "store") and hasattr(ws_tools.store, "query"):
                ws_tools.store.query(
                    "MATCH (f:Fact) WHERE f.block_id = $bid DETACH DELETE f",
                    {"bid": block_id},
                )

            await _broadcast_workspace(
                workspace_id,
                {"type": "prune_fact", "block_id": block_id},
            )
            return {"ok": True, "message": f"Fact {block_id} successfully pruned."}
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app


def serve(store_path: str, host: str = "127.0.0.1", port: int = 8080) -> None:  # pragma: no cover
    """Start the tool server backed by a graph store at ``store_path``."""
    from .graph_store_factory import get_graph_store
    from .graph_tool import GraphTools

    tools = GraphTools(get_graph_store(store_path))
    app = None

    try:
        import uvicorn

        app = make_fastapi_app(tools)
        print(
            f"Starting FastAPI server on http://{host}:{port}  (WS /ws, GET /tools, POST /tools/call)"
        )
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        from http.server import HTTPServer

        Handler = make_handler(tools)
        server = HTTPServer((host, port), Handler)
        print(f"[Fallback] Starting http.server on http://{host}:{port}  (GET /tools, POST /tools/call)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
    finally:
        if app is not None:
            for ws_tool in getattr(app.state, "workspace_tools", {}).values():
                ws_store = getattr(ws_tool, "store", None)
                if ws_store is not None and hasattr(ws_store, "close"):
                    try:
                        ws_store.close()
                    except Exception:
                        pass
        if hasattr(tools.store, "close"):
            try:
                tools.store.close()
            except Exception:
                pass
