from typing import Any, Dict, List, Optional

class BaseGraphStore:
    """Abstract base class defining the Graph Database interface."""

    def ingest_facts(self, facts: List[Dict[str, Any]]) -> int:
        raise NotImplementedError

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # --- Database-Agnostic Implementations based entirely on self.query ---

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
        """Identify pairs of facts with matching subject and object but contradictory predicates."""
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
        """Locate chains A -[RELATED]-> B and suggest transitive links."""
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
        """Perform path validation and demote contradictory facts."""
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
                self.query(
                    "MATCH (f:Fact) WHERE f.block_id = $bid SET f.reliability = 'UNVERIFIED'",
                    {"bid": c["a_id"]}
                )
                self.query(
                    "MATCH (f:Fact) WHERE f.block_id = $bid SET f.reliability = 'UNVERIFIED'",
                    {"bid": c["b_id"]}
                )
                demoted.append({"block_id": c["a_id"], "statement": c["a_stmt"]})
                demoted.append({"block_id": c["b_id"], "statement": c["b_stmt"]})
                continue

            self.query(
                "MATCH (f:Fact) WHERE f.block_id = $bid SET f.reliability = 'UNVERIFIED'",
                {"bid": target_id}
            )
            demoted.append({"block_id": target_id, "statement": target_stmt})

        return {"demoted": demoted}
