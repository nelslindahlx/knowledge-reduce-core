import hashlib
from typing import Any, Dict, List, Optional
from .graph_store_base import BaseGraphStore


def _block_id(fact: Dict[str, Any]) -> str:
    """Deterministic id from the SVO triple (idempotent re-ingest)."""
    key = "\x00".join(_s(fact.get(k)) for k in ("subject", "predicate", "object"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _s(v: Any) -> str:
    return str(v if v is not None else "")


class Neo4jStore(BaseGraphStore):
    """Neo4j cloud/server graph store of distilled facts (Cypher-queryable)."""

    def __init__(self, uri: str, user: str = "neo4j", password: str = "password", **kwargs):
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j library is missing. Install it with: pip install neo4j"
            )
        self._uri = uri
        self._user = kwargs.get("username", user)
        self._password = password
        self.driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self._init_constraints()

    def _init_constraints(self) -> None:
        """Create uniqueness constraints on Fact.block_id."""
        try:
            self.query(
                "CREATE CONSTRAINT fact_block_id_unique IF NOT EXISTS "
                "FOR (f:Fact) REQUIRE f.block_id IS UNIQUE"
            )
        except Exception as exc:
            print(f"[Neo4jStore] Constraint warning: {exc}")

    def ingest_facts(self, facts: List[Dict[str, Any]]) -> int:
        """Ingest a list of fact dictionaries. Returns count of new or merged facts."""
        if not facts:
            return 0

        query = (
            "MERGE (f:Fact {block_id: $bid}) "
            "ON CREATE SET "
            "  f.statement = $statement, "
            "  f.subject = $subject, "
            "  f.predicate = $predicate, "
            "  f.object = $object, "
            "  f.domain = $domain, "
            "  f.reliability = $reliability, "
            "  f.agreement = $agreement, "
            "  f.quality = $quality, "
            "  f.source_models = $source_models"
        )

        count = 0
        for f in facts:
            bid = _block_id(f)
            params = {
                "bid": bid,
                "statement": f.get("statement", ""),
                "subject": f.get("subject", ""),
                "predicate": f.get("predicate", ""),
                "object": f.get("object", ""),
                "domain": f.get("domain", ""),
                "reliability": _s(f.get("reliability")),
                "agreement": int(f.get("agreement", 1)),
                "quality": float(f.get("quality", 1.0)),
                "source_models": f.get("source_models", "")
            }
            self.query(query, params)
            count += 1
        return count

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the Neo4j database."""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def count(self) -> int:
        res = self.query("MATCH (f:Fact) RETURN count(f) AS n")
        return res[0]["n"] if res else 0

    def close(self) -> None:
        self.driver.close()
