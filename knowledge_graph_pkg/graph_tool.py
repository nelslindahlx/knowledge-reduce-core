"""
LLM-callable graph tools over a :class:`KuzuStore` (ModelReduce Session 6).

Wraps the graph store in a small, safe surface an LLM can call as functions
(directly, or via the MCP server in :mod:`mcp_server`). Read-only by design:
:meth:`GraphTools.graph_query` refuses any mutating/DDL Cypher and appends a
LIMIT when the caller omits one, so a model can explore the knowledge graph
without being able to corrupt it.

:data:`TOOL_SCHEMAS` are JSON-schema function definitions suitable for LLM
function-calling / MCP tool registration.
"""

import re
from typing import Any, Dict, List

# Cypher keywords that mutate data or schema -- forbidden in the tool layer.
_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|DROP|ALTER|COPY|INSTALL|LOAD|REMOVE)\b",
    re.IGNORECASE,
)


class GraphTools:
    """A read-only, LLM-facing facade over a KuzuStore."""

    def __init__(self, store: Any, default_limit: int = 100):
        self.store = store
        self.default_limit = default_limit

    def graph_query(self, cypher: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Run a READ-ONLY Cypher query. Mutating/DDL statements are refused.

        If the query has no LIMIT clause, one is appended (cap result size).
        """
        if _FORBIDDEN.search(cypher):
            raise ValueError(
                "graph_query is read-only; mutating/DDL Cypher is not allowed.")
        q = cypher.strip().rstrip(";")
        if "limit" not in q.lower():
            q += f" LIMIT {int(limit)}"
        return self.store.query(q)

    def graph_find_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        """Find facts whose subject matches ``subject`` (case-insensitive)."""
        return self.store.find_by_subject(subject)

    def graph_find_by_reliability(self, reliability: str,
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """Find facts at a given reliability tier, best-quality first."""
        return self.store.find_by_reliability(reliability, limit=limit)


# JSON-schema tool definitions for LLM function-calling / MCP registration.
TOOL_SCHEMAS = [
    {
        "name": "graph_query",
        "description": "Run a read-only Cypher query against the knowledge graph "
                       "of distilled, reliability-rated facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "cypher": {"type": "string", "description": "A read-only Cypher query."},
                "limit": {"type": "integer", "description": "Max rows (default 100)."},
            },
            "required": ["cypher"],
        },
    },
    {
        "name": "graph_find_by_subject",
        "description": "Find facts about a given subject (case-insensitive substring).",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject to search for."},
            },
            "required": ["subject"],
        },
    },
    {
        "name": "graph_find_by_reliability",
        "description": "List facts at a reliability tier "
                       "(UNVERIFIED|POSSIBLY_TRUE|LIKELY_TRUE|VERIFIED), best first.",
        "parameters": {
            "type": "object",
            "properties": {
                "reliability": {"type": "string"},
                "limit": {"type": "integer", "description": "Max rows (default 100)."},
            },
            "required": ["reliability"],
        },
    },
]
