import os
from typing import Any, Dict, List, Optional

from .graph_store_base import BaseGraphStore, normalize_fact_record


class Neo4jStore(BaseGraphStore):
    """Neo4j cloud/server graph store of distilled facts (Cypher-queryable)."""

    def __init__(
        self,
        uri: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        **kwargs,
    ):
        self._uri = uri
        self._user = kwargs.get("username") or user or os.environ.get("NEO4J_USER") or "neo4j"
        self._database = database

        existing_driver = kwargs.get("_driver")
        if existing_driver is not None:
            self.driver = existing_driver
            return

        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError(
                "neo4j library is missing. Install it with: pip install neo4j"
            ) from exc

        resolved_password = password if password is not None else os.environ.get("NEO4J_PASSWORD")
        if resolved_password is None:
            raise ValueError(
                "Neo4j password required via password argument or NEO4J_PASSWORD environment variable."
            )

        self.driver = GraphDatabase.driver(self._uri, auth=(self._user, resolved_password))
        self._init_constraints()

    def with_database(self, database: str) -> "Neo4jStore":
        """Create a workspace-scoped view sharing the existing Neo4j driver."""
        return Neo4jStore(
            self._uri,
            user=self._user,
            database=database,
            _driver=self.driver,
        )

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
            "SET "
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
            norm = normalize_fact_record(f)
            params = {
                "bid": norm["block_id"],
                "statement": norm["statement"],
                "subject": norm["subject"],
                "predicate": norm["predicate"],
                "object": norm["object"],
                "domain": norm["domain"],
                "reliability": norm["reliability"],
                "agreement": norm["agreement"],
                "quality": norm["quality"],
                "source_models": norm["source_models"],
            }
            self.query(query, params)
            count += 1
        return count

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the Neo4j database."""
        session_kwargs = {}
        if getattr(self, "_database", None):
            session_kwargs["database"] = self._database
        with self.driver.session(**session_kwargs) as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def count(self) -> int:
        res = self.query("MATCH (f:Fact) RETURN count(f) AS n")
        return res[0]["n"] if res else 0

    def close(self) -> None:
        self.driver.close()
