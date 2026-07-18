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
    <script src="https://unpkg.com/3d-force-graph"></script>
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
        let Graph = null;
        let allNodes = [];
        let allEdges = [];

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
            const visLinks = [];
            const subjectsAndObjects = new Map();

            nodes.forEach(f => {
                if (f.subject) {
                    const sLower = f.subject.trim().toLowerCase();
                    if (!subjectsAndObjects.has(sLower)) {
                        subjectsAndObjects.set(sLower, { 
                            id: 'concept_' + sLower, 
                            label: f.subject.trim(), 
                            type: 'concept',
                            semType: f.subject_type || 'CONCEPT',
                            parentClass: f.subject_parent || ''
                        });
                    }
                }
                if (f.object) {
                    const oLower = f.object.trim().toLowerCase();
                    if (!subjectsAndObjects.has(oLower)) {
                        subjectsAndObjects.set(oLower, { 
                            id: 'concept_' + oLower, 
                            label: f.object.trim(), 
                            type: 'concept',
                            semType: f.object_type || 'CONCEPT',
                            parentClass: f.object_parent || ''
                        });
                    }
                }
            });

            const typeColors = {
                'PROCESS': '#10b981', // Emerald
                'ENTITY': '#3b82f6',   // Blue
                'LOCATION': '#f59e0b', // Amber
                'ATTRIBUTE': '#ec4899',// Pink
                'CONCEPT': '#06b6d4'   // Cyan
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

            nodes.forEach((f, idx) => {
                if (f.subject && f.object) {
                    const fromId = 'concept_' + f.subject.trim().toLowerCase();
                    const toId = 'concept_' + f.object.trim().toLowerCase();
                    const relColor = reliabilityColors[f.reliability] || '#9ca3af';
                    
                    visLinks.push({
                        id: 'fact_' + f.block_id,
                        source: fromId,
                        target: toId,
                        label: `${f.predicate} (${f.reliability})`,
                        color: relColor,
                        fact: f
                    });
                }
            });

            const container = document.getElementById('mynetwork');
            container.innerHTML = '';
            
            Graph = ForceGraph3D()(container)
                .graphData({ nodes: visNodes, links: visLinks })
                .nodeColor(node => node.color)
                .nodeLabel(node => `<div class="node-tooltip"><b>${node.label}</b></div>`)
                .nodeVal(node => node.val)
                .linkColor(link => link.color)
                .linkLabel(link => `<div class="link-tooltip"><b>${link.label}</b><br/>${link.fact.statement}</div>`)
                .linkDirectionalArrowLength(4)
                .linkDirectionalArrowRelPos(0.95)
                .linkCurvature(0.15)
                .backgroundColor('#0b0f19')
                .onNodeClick(node => {
                    const distance = 80;
                    const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
                    Graph.cameraPosition(
                        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
                        node,
                        1500
                    );
                    
                    const conceptName = node.id.replace('concept_', '');
                    const relatedFacts = nodes.filter(n => 
                        (n.subject && n.subject.trim().toLowerCase() === conceptName) ||
                        (n.object && n.object.trim().toLowerCase() === conceptName)
                    );
                    showConceptDetails(conceptName, relatedFacts);
                })
                .onLinkClick(link => {
                    showFactDetails(link.fact);
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
                <div class="detail-item" style="margin-top:16px;">
                    <button class="prune-btn" onclick="pruneFact('${fact.block_id}')">Prune Fact</button>
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

        async function pruneFact(blockId) {
            if (!confirm("Are you sure you want to prune this fact from the knowledge graph?")) {
                return;
            }
            try {
                const response = await fetch('/api/facts/' + blockId, { method: 'DELETE' });
                const resData = await response.json();
                if (resData.ok) {
                    edgesDataset.remove('fact_' + blockId);
                    allNodes = allNodes.filter(n => n.block_id !== blockId);
                    updateStats(allNodes);
                    document.getElementById('detail-card').innerHTML = '<div style="color:var(--text-muted); font-size:14px; font-style:italic; text-align:center; margin-top:40px;">Fact pruned successfully. Select another fact or node to view details.</div>';
                } else {
                    alert("Error pruning fact: " + resData.error);
                }
            } catch (err) {
                console.error("Pruning failed:", err);
                alert("Pruning failed: " + err);
            }
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

        function setupWebSocket() {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                const ws = new WebSocket(wsUrl);

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
                            renderNetwork(allNodes, allEdges);
                            updateStats(allNodes);
                        } else if (update.type === 'prune_fact') {
                            const blockId = update.block_id;
                            allNodes = allNodes.filter(n => n.block_id !== blockId);
                            renderNetwork(allNodes, allEdges);
                            updateStats(allNodes);
                        }
                    } catch (e) {
                        console.error("Error processing websocket update:", e);
                    }
                };

                ws.onclose = () => {
                    setTimeout(setupWebSocket, 5000);
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


def make_fastapi_app(tools: Any) -> Any:
    """Create a FastAPI application serving the visual dashboard and tool endpoints."""
    import os
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Security, Depends
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from typing import Set, Dict, Optional
    
    app = FastAPI(title="KnowledgeReduce API Gateway")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    security = HTTPBearer(auto_error=False)
    workspace_tools: Dict[str, Any] = {}

    async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
        secret = os.environ.get("MCP_JWT_SECRET")
        if not secret:
            return
        if not credentials or credentials.credentials != secret:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing bearer token")

    def get_tools_for_workspace(workspace_id: str) -> Any:
        if not workspace_id or workspace_id == "default":
            return tools
        if workspace_id not in workspace_tools:
            from .graph_store_factory import get_graph_store
            from .graph_tool import GraphTools
            from .neo4j_store import Neo4jStore
            
            if isinstance(tools.store, Neo4jStore):
                ws_path = f"{tools.store._uri}/{workspace_id}"
            else:
                db_path = getattr(tools.store, "db_path", None)
                if not isinstance(db_path, str):
                    db_path = getattr(tools.store, "_uri", "graph_db")
                if not isinstance(db_path, str):
                    db_path = "graph_db"
                    
                if "/" in db_path or "\\" in db_path or db_path == "local_path":
                    ws_path = os.path.join(os.path.dirname(db_path), f"graph_db_{workspace_id}")
                else:
                    ws_path = f"{db_path}_{workspace_id}"
                
            workspace_tools[workspace_id] = GraphTools(get_graph_store(ws_path))
        return workspace_tools[workspace_id]

    def get_workspace_id(request: Request) -> str:
        return request.headers.get("x-workspace-id", "default")

    active_websockets: Set[WebSocket] = set()

    @app.get("/tools", dependencies=[Depends(verify_token)])
    async def get_tools():
        return {"tools": list_tools()}

    @app.post("/tools/call", dependencies=[Depends(verify_token)])
    async def post_tools_call(request: Request):
        try:
            req = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"ok": False, "error": "invalid JSON"})
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
            sem_types = distiller.infer_semantic_types()
            taxonomy = distiller.distill_taxonomy()
            
            # Map child back to parent class
            parent_classes = {}
            for parent, children in taxonomy.items():
                for child in children:
                    parent_classes[child.lower()] = parent
                    
            enriched_nodes = []
            for n in nodes:
                s = str(n.get("subject", ""))
                o = str(n.get("object", ""))
                node_copy = dict(n)
                node_copy["subject_type"] = sem_types.get(s, "CONCEPT")
                node_copy["object_type"] = sem_types.get(o, "CONCEPT")
                node_copy["subject_parent"] = parent_classes.get(s.lower(), None)
                node_copy["object_parent"] = parent_classes.get(o.lower(), None)
                enriched_nodes.append(node_copy)
                
            return {"nodes": enriched_nodes, "edges": edges}
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})


    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        active_websockets.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            active_websockets.discard(websocket)

    @app.post("/api/notify_update")
    async def notify_update(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "invalid JSON"})
            
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        
        closed_sockets = set()
        for ws in list(active_websockets):
            try:
                await ws.send_json({
                    "type": "graph_update",
                    "nodes": nodes,
                    "edges": edges
                })
            except Exception:
                closed_sockets.add(ws)
                
        for ws in closed_sockets:
            active_websockets.discard(ws)
            
        return {"status": "broadcasted", "recipients": len(active_websockets)}

    @app.delete("/api/facts/{block_id}", dependencies=[Depends(verify_token)])
    async def delete_fact(block_id: str, request: Request):
        try:
            workspace_id = get_workspace_id(request)
            ws_tools = get_tools_for_workspace(workspace_id)
            if hasattr(ws_tools, "store") and hasattr(ws_tools.store, "query"):
                ws_tools.store.query("MATCH (f:Fact) WHERE f.block_id = $bid DETACH DELETE f", {"bid": block_id})
            
            # Broadcast the pruning event to active WebSocket visual clients
            closed_sockets = set()
            for ws in list(active_websockets):
                try:
                    await ws.send_json({
                        "type": "prune_fact",
                        "block_id": block_id
                    })
                except Exception:
                    closed_sockets.add(ws)
            for ws in closed_sockets:
                active_websockets.discard(ws)
                
            return {"ok": True, "message": f"Fact {block_id} successfully pruned."}
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app



def serve(store_path: str, host: str = "127.0.0.1", port: int = 8080) -> None:  # pragma: no cover
    """Start the tool server backed by a graph store at ``store_path``."""
    from .graph_store_factory import get_graph_store
    from .graph_tool import GraphTools

    tools = GraphTools(get_graph_store(store_path))
    
    try:
        import uvicorn
        app = make_fastapi_app(tools)
        print(f"Starting FastAPI server on http://{host}:{port}  (WS /ws, GET /tools, POST /tools/call)")
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
