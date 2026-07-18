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


def _block_id(fact: Dict[str, Any]) -> str:
    """Deterministic id from the SVO triple (idempotent re-ingest)."""
    key = "\x00".join(_s(fact.get(k)) for k in ("subject", "predicate", "object"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _s(v: Any) -> str:
    return str(v if v is not None else "")


class KuzuStore:
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

    def find_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        """Facts whose subject contains ``subject`` (case-insensitive)."""
        return self.query(
            "MATCH (f:Fact) WHERE lower(f.subject) CONTAINS lower($s) "
            "RETURN f.subject AS subject, f.predicate AS predicate, "
            "f.object AS object, f.reliability AS reliability, "
            "f.statement AS statement",
            {"s": subject},
        )

    def find_by_reliability(self, reliability: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self.query(
            "MATCH (f:Fact) WHERE f.reliability = $r "
            "RETURN f.statement AS statement, f.subject AS subject, "
            "f.object AS object ORDER BY f.quality DESC LIMIT " + str(int(limit)),
            {"r": reliability},
        )

    def auto_link_relations(self) -> int:
        """Create RELATED edges where Fact a's object matches Fact b's subject.

        Is case-insensitive and avoids self-loops. Returns count of links created.
        """
        query = (
            "MATCH (a:Fact), (b:Fact) "
            "WHERE lower(a.object) = lower(b.subject) AND a.block_id <> b.block_id "
            "MERGE (a)-[r:RELATED]->(b) "
            "SET r.predicate = a.predicate + '_' + b.predicate, r.weight = 1.0 "
            "RETURN count(r) AS num_links"
        )
        res = self.query(query)
        return res[0]["num_links"] if res else 0

    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Identify pairs of facts with matching subject and object but contradictory predicates.

        A contradiction is identified if one predicate contains a negation word
        (not, never, no, doesn't, cannot) and the other does not.
        """
        pairs = self.query(
            "MATCH (a:Fact), (b:Fact) "
            "WHERE lower(a.subject) = lower(b.subject) AND lower(a.object) = lower(b.object) "
            "AND a.block_id < b.block_id "
            "RETURN a.block_id AS a_id, a.statement AS a_stmt, a.predicate AS a_pred, a.reliability AS a_rel, a.quality AS a_qual, "
            "b.block_id AS b_id, b.statement AS b_stmt, b.predicate AS b_pred, b.reliability AS b_rel, b.quality AS b_qual"
        )

        negations = {"not", "never", "no", "doesn't", "doesnt", "cannot", "cant", "can't", "fails"}
        contradictions = []
        for p in pairs:
            p1_words = set(p["a_pred"].lower().split())
            p2_words = set(p["b_pred"].lower().split())
            has_neg1 = bool(p1_words & negations)
            has_neg2 = bool(p2_words & negations)
            if has_neg1 != has_neg2:
                contradictions.append(p)
        return contradictions

    def find_transitive_inferences(self) -> List[Dict[str, Any]]:
        """Locate chains A -[RELATED]-> B and suggest transitive links.

        For example: A -> is -> B, B -> is -> C implies A -> C.
        """
        res = self.query(
            "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
            "WHERE a.predicate = b.predicate "
            "RETURN a.subject AS subject, a.predicate AS predicate, b.object AS object, "
            "a.statement AS step1, b.statement AS step2"
        )
        seen = set()
        inferences = []
        for row in res:
            key = (row["subject"].lower(), row["predicate"].lower(), row["object"].lower())
            if key not in seen:
                seen.add(key)
                inferences.append(row)
        return inferences

    def validate_and_reconcile(self) -> Dict[str, Any]:
        """Perform path validation and demote contradictory facts.

        For any contradictory pair, the fact with the lower quality score is
        demoted to 'UNVERIFIED'. Returns a list of demoted block IDs and statements.
        """
        contradictions = self.find_contradictions()
        demoted = []
        for c in contradictions:
            if c["a_rel"] == "UNVERIFIED" or c["b_rel"] == "UNVERIFIED":
                continue

            if c["a_qual"] < c["b_qual"]:
                target_id = c["a_id"]
                target_stmt = c["a_stmt"]
            elif c["b_qual"] < c["a_qual"]:
                target_id = c["b_id"]
                target_stmt = c["b_stmt"]
            else:
                self._execute(
                    "MATCH (f:Fact) WHERE f.block_id = $bid SET f.reliability = 'UNVERIFIED'",
                    {"bid": c["a_id"]}
                )
                self._execute(
                    "MATCH (f:Fact) WHERE f.block_id = $bid SET f.reliability = 'UNVERIFIED'",
                    {"bid": c["b_id"]}
                )
                demoted.append({"block_id": c["a_id"], "statement": c["a_stmt"]})
                demoted.append({"block_id": c["b_id"], "statement": c["b_stmt"]})
                continue

            self._execute(
                "MATCH (f:Fact) WHERE f.block_id = $bid SET f.reliability = 'UNVERIFIED'",
                {"bid": target_id}
            )
            demoted.append({"block_id": target_id, "statement": target_stmt})

        return {"demoted": demoted}

    def close(self) -> None:
        # Drop references so KùzuDB releases the on-disk lock for reopen.
        self.conn = None
        self.db = None
