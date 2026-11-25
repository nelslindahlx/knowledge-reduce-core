"""
LLM-callable graph tools over a :class:`BaseGraphStore` (ModelReduce Session 6).

Wraps the graph store in a small, safe surface an LLM can call as functions
(directly, or via the MCP server in :mod:`mcp_server`). Read-only by design:
:meth:`GraphTools.graph_query` refuses any mutating/DDL Cypher, multi-statement
semicolons, and comments, and appends a LIMIT when the caller omits one, so a
model can explore the knowledge graph without being able to corrupt it.

:data:`TOOL_SCHEMAS` are JSON-schema function definitions suitable for LLM
function-calling / MCP tool registration.
"""

import re
from typing import Any, Dict, List, Optional

from .rag import GraphRAGRetriever

# Cypher keywords that mutate data, schema, procedures, or session state.
_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|DROP|ALTER|COPY|INSTALL|LOAD|REMOVE|"
    r"CALL|FOREACH|ATTACH|EXPORT|IMPORT|USE|GRANT|DENY|REVOKE|CHECKPOINT|"
    r"COMMENT|BEGIN|COMMIT|ROLLBACK|TERMINATE|RENAME)\b",
    re.IGNORECASE,
)


def _strip_cypher_literals(cypher: str) -> str:
    """Replace single- and double-quoted string literals with empty placeholders."""
    out: List[str] = []
    i = 0
    n = len(cypher)
    while i < n:
        ch = cypher[i]
        if ch in ("'", '"'):
            quote = ch
            out.append(quote)
            i += 1
            while i < n:
                cur = cypher[i]
                if cur == "\\":
                    i += 2
                    continue
                if cur == quote:
                    if i + 1 < n and cypher[i + 1] == quote:
                        i += 2
                        continue
                    out.append(quote)
                    i += 1
                    break
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


class GraphTools:
    """A read-only, LLM-facing facade over a BaseGraphStore."""

    def __init__(self, store: Any, default_limit: int = 100):
        self.store = store
        self.default_limit = max(0, int(default_limit))
        self._retriever: Optional[GraphRAGRetriever] = None

    def _get_retriever(self) -> GraphRAGRetriever:
        if self._retriever is None:
            self._retriever = GraphRAGRetriever(self.store)
        return self._retriever

    def graph_query(self, cypher: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run a READ-ONLY Cypher query. Mutating/DDL statements are refused.

        If the query has no LIMIT clause, one is appended (cap result size).
        """
        if not isinstance(cypher, str) or not cypher.strip():
            raise ValueError("cypher query must be a non-empty string.")

        stripped = _strip_cypher_literals(cypher.strip())
        if "//" in stripped or "/*" in stripped:
            raise ValueError("graph_query does not allow Cypher comments.")
        if ";" in stripped.rstrip(";"):
            raise ValueError("graph_query allows only a single Cypher statement.")
        if _FORBIDDEN.search(stripped):
            raise ValueError(
                "graph_query is read-only; mutating/DDL Cypher is not allowed."
            )

        effective_limit = self.default_limit if limit is None else max(0, int(limit))
        q = cypher.strip().rstrip(";")
        if not re.search(r"\bLIMIT\b", _strip_cypher_literals(q), re.IGNORECASE):
            q += f" LIMIT {effective_limit}"
        rows = self.store.query(q)
        return list(rows)[:effective_limit] if isinstance(rows, list) else rows

    def graph_find_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        """Find facts whose subject matches ``subject`` (case-insensitive)."""
        return self.store.find_by_subject(subject)

    def graph_find_by_reliability(
        self, reliability: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find facts at a given reliability tier, best-quality first."""
        effective_limit = self.default_limit if limit is None else max(0, int(limit))
        return self.store.find_by_reliability(reliability, limit=effective_limit)

    def graph_rag_retrieve(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 2,
        exclude_unverified: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid multi-hop Graph-RAG retrieval over the knowledge store.

        Returns semantically relevant facts and their adjacent concept hops, scored
        by semantic similarity and PageRank network importance.
        """
        retriever = self._get_retriever()
        return retriever.retrieve(
            query=query,
            top_k=top_k,
            hops=hops,
            exclude_unverified=exclude_unverified,
        )

    def graph_distill_ontology(self) -> Dict[str, Any]:
        """Distill the high-level taxonomy, semantic categories, and relation schema from the graph."""
        from .ontology import OntologyDistiller

        distiller = OntologyDistiller(self.store)
        return distiller.distill_all()


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
    {
        "name": "graph_rag_retrieve",
        "description": "Perform a hybrid multi-hop Graph-RAG retrieval over the knowledge graph. "
        "Retrieves semantically similar facts and walks their relationship paths.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query string to retrieve context for."},
                "top_k": {
                    "type": "integer",
                    "description": "Number of initial seed facts to retrieve (default: 5).",
                },
                "hops": {
                    "type": "integer",
                    "description": "Number of graph connection hops to walk (default: 2).",
                },
                "exclude_unverified": {
                    "type": "boolean",
                    "description": "Exclude UNVERIFIED/contradicted facts (default: true).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "graph_distill_ontology",
        "description": "Distill the high-level taxonomy, semantic categories, and relation schema from the graph.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]
