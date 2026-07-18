"""
KùzuDB graph store for ModelReduce (Session 6).

Distilled facts are richer than flat JSONL: they have subjects, objects, and
(eventually) relationships between them. :class:`KuzuStore` loads facts into
an embedded property graph (KùzuDB -- "the SQLite of graph DBs") so they can
be queried with **standard Cypher** and traversed, without standing up a
server or hand-writing a query parser.

kuzu is an optional dependency (the ``graph`` extra) and is imported lazily,
so the package works without it; only constructing a :class:`KuzuStore`
requires it.

Schema (v1): a single ``Fact`` node table keyed by a deterministic
``block_id`` (so re-ingesting the same fact is idempotent). A ``RELATED`` rel
table is created for future fact-to-fact edges.
"""

import hashlib
from typing import Any, Dict, List, Optional


from .graph_store_base import BaseGraphStore


def _block_id(fact: Dict[str, Any]) -> str:
    """Deterministic id from the SVO triple (idempotent re-ingest)."""
    key = "\x00".join(_s(fact.get(k)) for k in ("subject", "predicate", "object"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _s(v: Any) -> str:
    return str(v if v is not None else "")


class KuzuStore(BaseGraphStore):
    """Embedded property-graph store of distilled facts (Cypher-queryable)."""

    def __init__(self, path: str):
        try:
            import kuzu
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise ImportError(
                "KuzuStore requires the graph extra: pip install knowledgereduce[graph]"
            ) from exc
        self._kuzu = kuzu
        self.path = path
        
        import time
        max_retries = 5
        backoff = 0.1
        for attempt in range(max_retries):
            try:
                self.db = kuzu.Database(path)
                self.conn = kuzu.Connection(self.db)
                break
            except RuntimeError as exc:
                if attempt == max_retries - 1:
                    raise exc
                time.sleep(backoff)
                backoff *= 2
                
        self._init_schema()

    def _execute(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        """Execute a query with auto-retry on lock errors."""
        import time
        max_retries = 5
        backoff = 0.1
        for attempt in range(max_retries):
            try:
                return self.conn.execute(cypher, params or {})
            except RuntimeError as exc:
                err_msg = str(exc).lower()
                if "lock" in err_msg or "connection" in err_msg or attempt == max_retries - 1:
                    if attempt == max_retries - 1:
                        raise exc
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise exc

    def _init_schema(self) -> None:
        # CREATE ... IF NOT EXISTS keeps reopen/persist working.
        self._execute(
            "CREATE NODE TABLE IF NOT EXISTS Fact("
            "block_id STRING, statement STRING, subject STRING, predicate STRING, "
            "object STRING, domain STRING, reliability STRING, agreement INT64, "
            "quality INT64, source_models STRING, PRIMARY KEY(block_id))"
        )
        self._execute(
            "CREATE REL TABLE IF NOT EXISTS RELATED(FROM Fact TO Fact, "
            "predicate STRING, weight DOUBLE)"
        )

    # ------------------------------------------------------------------ #
    def ingest_facts(self, facts: List[Dict[str, Any]]) -> int:
        """Insert facts as Fact nodes. Idempotent by block_id. Returns count seen."""
        for f in facts:
            bid = _block_id(f)
            models = f.get("source_models") or []
            params = {
                "bid": bid,
                "stmt": _s(f.get("fact_statement")),
                "subj": _s(f.get("subject")),
                "pred": _s(f.get("predicate")),
                "obj": _s(f.get("object")),
                "domain": _s(f.get("domain") or f.get("category")),
                "rel": _s(f.get("reliability_rating")),
                "agree": int(f.get("cross_model_agreement", 1) or 1),
                "quality": int(f.get("quality_score", 0) or 0),
                "models": ", ".join(models) if isinstance(models, list) else _s(models),
            }
            # MERGE makes re-ingest idempotent (no duplicate block_id).
            self._execute(
                "MERGE (f:Fact {block_id: $bid}) "
                "SET f.statement = $stmt, f.subject = $subj, f.predicate = $pred, "
                "f.object = $obj, f.domain = $domain, f.reliability = $rel, "
                "f.agreement = $agree, f.quality = $quality, f.source_models = $models",
                params,
            )
        return len(facts)

    # ------------------------------------------------------------------ #
    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a Cypher query; return a list of column-name -> value dicts."""
        result = self._execute(cypher, params or {})
        cols = result.get_column_names()
        rows: List[Dict[str, Any]] = []
        while result.has_next():
            vals = result.get_next()
            rows.append({c: v for c, v in zip(cols, vals)})
        return rows

    def count(self) -> int:
        return self.query("MATCH (f:Fact) RETURN count(f) AS n")[0]["n"]

    def close(self) -> None:
        # Drop references so KùzuDB releases the on-disk lock for reopen.
        self.conn = None
        self.db = None
