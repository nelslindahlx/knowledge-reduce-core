from enum import Enum
import hashlib
import re
from typing import Any, Dict, List, Optional, Set

RELIABILITY_RANK: Dict[str, int] = {
    "UNVERIFIED": 0,
    "POSSIBLY_TRUE": 1,
    "LIKELY_TRUE": 2,
    "VERIFIED": 3,
}

_NEGATION_TOKENS: Set[str] = {
    "not",
    "never",
    "no",
    "doesn't",
    "doesnt",
    "don't",
    "dont",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "won't",
    "wont",
    "cannot",
    "cant",
    "can't",
    "fails",
}

_CONTRADICTION_AUX: Set[str] = {
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "will",
    "can",
    "to",
}


def _s(v: Any) -> str:
    """Convert optional values cleanly to strings without turning None into 'None'."""
    return "" if v is None else str(v)


def _block_id(fact: Dict[str, Any]) -> str:
    """Deterministic id from the SVO triple (idempotent re-ingest)."""
    key = "\x00".join(_s(fact.get(k)) for k in ("subject", "predicate", "object"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def normalize_fact_record(fact: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize canonical and legacy fact dictionaries into a uniform store record."""
    stmt = (
        fact.get("fact_statement")
        if fact.get("fact_statement") is not None
        else fact.get("statement")
    )
    domain = fact.get("domain") if fact.get("domain") is not None else fact.get("category")

    raw_rel = (
        fact.get("reliability_rating")
        if fact.get("reliability_rating") is not None
        else fact.get("reliability")
    )
    if isinstance(raw_rel, Enum):
        reliability = raw_rel.name
    else:
        rel_str = _s(raw_rel).strip()
        reliability = rel_str.upper() if rel_str.upper() in RELIABILITY_RANK else rel_str

    raw_agree = (
        fact.get("cross_model_agreement")
        if fact.get("cross_model_agreement") is not None
        else fact.get("agreement")
    )
    agreement = int(raw_agree) if raw_agree is not None else 1

    raw_qual = (
        fact.get("quality_score")
        if fact.get("quality_score") is not None
        else fact.get("quality")
    )
    quality = round(float(raw_qual)) if raw_qual is not None else 0

    models = fact.get("source_models")
    if isinstance(models, (list, tuple)):
        source_models = ", ".join(_s(m) for m in models)
    else:
        source_models = _s(models)

    return {
        "block_id": _block_id(fact),
        "statement": _s(stmt),
        "subject": _s(fact.get("subject")),
        "predicate": _s(fact.get("predicate")),
        "object": _s(fact.get("object")),
        "domain": _s(domain),
        "reliability": reliability,
        "agreement": agreement,
        "quality": quality,
        "source_models": source_models,
    }


def _words_compatible(w1: str, w2: str) -> bool:
    if w1 == w2:
        return True
    if min(len(w1), len(w2)) >= 4 and (w1.startswith(w2) or w2.startswith(w1)):
        return True
    return False


def _core_predicates_match(pred1: str, pred2: str) -> bool:
    tokens1 = set(re.findall(r"[a-z0-9']+", pred1.lower()))
    tokens2 = set(re.findall(r"[a-z0-9']+", pred2.lower()))
    has_neg1 = bool(tokens1 & _NEGATION_TOKENS)
    has_neg2 = bool(tokens2 & _NEGATION_TOKENS)
    if has_neg1 == has_neg2:
        return False

    core1 = tokens1 - _NEGATION_TOKENS - _CONTRADICTION_AUX
    core2 = tokens2 - _NEGATION_TOKENS - _CONTRADICTION_AUX
    if not core1 and not core2:
        return True
    if not core1 or not core2:
        return False
    return any(_words_compatible(w1, w2) for w1 in core1 for w2 in core2)


class BaseGraphStore:
    """Abstract base class defining the Graph Database interface."""

    def ingest_facts(self, facts: List[Dict[str, Any]]) -> int:
        raise NotImplementedError

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def edge_count(self) -> int:
        """Return the total number of RELATED edges in the graph store."""
        res = self.query("MATCH ()-[r:RELATED]->() RETURN count(r) AS n")
        return int(res[0]["n"]) if res else 0

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
        safe_limit = max(0, int(limit))
        return self.query(
            "MATCH (f:Fact) WHERE f.reliability = $r "
            "RETURN f.statement AS statement, f.subject AS subject, "
            "f.object AS object ORDER BY f.quality DESC LIMIT " + str(safe_limit),
            {"r": reliability},
        )

    def auto_link_relations(self) -> int:
        """Create RELATED edges where Fact a's object matches Fact b's subject.

        Is case-insensitive and avoids self-loops or empty concept joins.
        Returns count of links present.
        """
        query = (
            "MATCH (a:Fact), (b:Fact) "
            "WHERE a.object <> '' AND b.subject <> '' "
            "AND lower(a.object) = lower(b.subject) AND a.block_id <> b.block_id "
            "MERGE (a)-[r:RELATED]->(b) "
            "ON CREATE SET r.predicate = coalesce(a.predicate, '') + '_' + coalesce(b.predicate, ''), "
            "r.weight = 1.0 "
            "RETURN count(r) AS num_links"
        )
        res = self.query(query)
        return res[0]["num_links"] if res else 0

    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Identify pairs of facts with matching subject and object but contradictory predicates."""
        pairs = self.query(
            "MATCH (a:Fact), (b:Fact) "
            "WHERE a.subject <> '' AND a.object <> '' "
            "AND lower(a.subject) = lower(b.subject) AND lower(a.object) = lower(b.object) "
            "AND a.block_id < b.block_id "
            "RETURN a.block_id AS a_id, a.statement AS a_stmt, a.predicate AS a_pred, "
            "a.reliability AS a_rel, a.quality AS a_qual, "
            "b.block_id AS b_id, b.statement AS b_stmt, b.predicate AS b_pred, "
            "b.reliability AS b_rel, b.quality AS b_qual"
        )
        contradictions = []
        for p in pairs:
            pred1 = _s(p.get("a_pred"))
            pred2 = _s(p.get("b_pred"))
            if _core_predicates_match(pred1, pred2):
                contradictions.append(p)
        return contradictions

    def find_transitive_inferences(self) -> List[Dict[str, Any]]:
        """Locate verified chains A -[RELATED]-> B and suggest transitive links."""
        res = self.query(
            "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
            "WHERE lower(a.predicate) = lower(b.predicate) "
            "AND a.reliability <> 'UNVERIFIED' AND b.reliability <> 'UNVERIFIED' "
            "AND lower(a.subject) <> lower(b.object) "
            "RETURN a.subject AS subject, a.predicate AS predicate, b.object AS object, "
            "a.statement AS step1, b.statement AS step2"
        )
        seen = set()
        inferences = []
        for row in res:
            subj = _s(row.get("subject"))
            pred = _s(row.get("predicate"))
            obj = _s(row.get("object"))
            if not subj or not pred or not obj:
                continue
            key = (subj.lower(), pred.lower(), obj.lower())
            if key not in seen:
                seen.add(key)
                inferences.append(row)
        return inferences

    def validate_and_reconcile(self) -> Dict[str, Any]:
        """Perform path validation and demote contradictory facts."""
        contradictions = self.find_contradictions()
        demoted = []
        demoted_ids: Set[str] = set()

        for c in contradictions:
            a_id = c["a_id"]
            b_id = c["b_id"]
            a_rel = _s(c.get("a_rel")).upper()
            b_rel = _s(c.get("b_rel")).upper()
            if (
                a_id in demoted_ids
                or b_id in demoted_ids
                or a_rel == "UNVERIFIED"
                or b_rel == "UNVERIFIED"
            ):
                continue

            a_score = (RELIABILITY_RANK.get(a_rel, 0), int(c.get("a_qual") or 0))
            b_score = (RELIABILITY_RANK.get(b_rel, 0), int(c.get("b_qual") or 0))

            if a_score < b_score:
                targets = [(a_id, c.get("a_stmt"))]
            elif b_score < a_score:
                targets = [(b_id, c.get("b_stmt"))]
            else:
                targets = [(a_id, c.get("a_stmt")), (b_id, c.get("b_stmt"))]

            for target_id, target_stmt in targets:
                self.query(
                    "MATCH (f:Fact) WHERE f.block_id = $bid SET f.reliability = 'UNVERIFIED'",
                    {"bid": target_id},
                )
                demoted_ids.add(target_id)
                demoted.append({"block_id": target_id, "statement": target_stmt})

        return {"demoted": demoted}
