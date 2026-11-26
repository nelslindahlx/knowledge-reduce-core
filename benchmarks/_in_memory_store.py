"""Shared offline helpers (InMemoryGraphStore & chat JSONL compiler) for benchmarks."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from knowledge_graph_pkg.core import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.distillation import KnowledgeDistiller
from knowledge_graph_pkg.graph_store_base import (
    BaseGraphStore,
    _block_id,
    _s,
    normalize_fact_record,
)


def chat_jsonl_from_facts(facts: list[dict[str, Any]]) -> list[str]:
    """Render stored fact records to deterministic chat JSONL lines via KnowledgeDistiller."""
    kg = KnowledgeGraph()
    fixed_ts = datetime(2026, 1, 1)
    for fact in facts:
        fid = str(fact["fact_id"])
        raw_tags = fact.get("tags") or []
        tags = (
            [t.strip() for t in raw_tags.split(",") if t.strip()]
            if isinstance(raw_tags, str)
            else list(raw_tags)
        )
        rel_name = str(fact.get("reliability_rating") or "LIKELY_TRUE").upper()
        rel = ReliabilityRating.__members__.get(rel_name, ReliabilityRating.LIKELY_TRUE)
        kg.add_fact(
            fact_id=fid,
            fact_statement=str(fact["fact_statement"]),
            category=str(fact.get("category") or "General"),
            tags=tags,
            date_recorded=fixed_ts,
            last_updated=fixed_ts,
            reliability_rating=rel,
            source_id=str(fact.get("source_id") or "benchmark"),
            source_title=str(fact.get("source_title") or "Benchmark"),
            author_creator=str(fact.get("author_creator") or "SVOExtractor"),
            publication_date=fixed_ts,
            url_reference=str(fact.get("url_reference") or ""),
            related_facts=[],
            contextual_notes=str(fact.get("contextual_notes") or ""),
            access_level=str(fact.get("access_level") or "public"),
            usage_count=int(fact.get("usage_count") or 1),
        )
        if fact.get("question"):
            kg.graph.nodes[fid]["question"] = str(fact["question"])
        if fact.get("answer"):
            kg.graph.nodes[fid]["answer"] = str(fact["answer"])
        if fact.get("quality_score") is not None:
            kg.graph.nodes[fid]["quality_score"] = int(fact["quality_score"])
    content = KnowledgeDistiller(kg).to_chat_jsonl()
    return [ln for ln in content.splitlines() if ln.strip()]


_PROJ_RE = re.compile(r"([a-z])\.([a-z_]+)\s+AS\s+([a-z0-9_]+)", re.IGNORECASE)


def _project(bindings: dict[str, dict[str, Any]], cypher: str) -> dict[str, Any]:
    """Project aliased fields from ``RETURN var.field AS alias``."""
    ret_idx = cypher.upper().find("RETURN ")
    ret_clause = cypher[ret_idx + 7 :] if ret_idx >= 0 else ""
    out: dict[str, Any] = {}
    for var, field, alias in _PROJ_RE.findall(ret_clause):
        rec = bindings.get(var.lower(), {})
        out[alias] = rec.get(field)
    return out


class InMemoryGraphStore(BaseGraphStore):
    """Deterministic in-memory :class:`BaseGraphStore` supporting core Cypher queries."""

    def __init__(self) -> None:
        self._facts: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str, str, float]] = []

    def ingest_facts(self, facts: list[dict[str, Any]]) -> int:
        """Normalize and upsert *facts* by deterministic ``block_id``."""
        count = 0
        for raw in facts:
            rec = normalize_fact_record(raw)
            bid = rec.get("block_id") or _block_id(rec)
            rec["block_id"] = bid
            self._facts[bid] = rec
            count += 1
        return count

    def count(self) -> int:
        return len(self._facts)

    def edge_count(self) -> int:
        return len(self._edges)

    def close(self) -> None:
        return None

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute supported read/write Cypher patterns in memory."""
        q = " ".join(cypher.split())
        q_upper = q.upper()
        p = params or {}

        if "SET F.RELIABILITY = 'UNVERIFIED'" in q_upper:
            bid = _s(p.get("bid"))
            if bid in self._facts:
                self._facts[bid]["reliability"] = "UNVERIFIED"
            return []

        if "MERGE (A)-[R:RELATED]->(B)" in q_upper:
            existing = {(src, dst) for src, dst, _, _ in self._edges}
            items = list(self._facts.values())
            for a in items:
                a_obj = _s(a.get("object"))
                if not a_obj:
                    continue
                for b in items:
                    b_subj = _s(b.get("subject"))
                    if (
                        not b_subj
                        or a["block_id"] == b["block_id"]
                        or a_obj.lower() != b_subj.lower()
                    ):
                        continue
                    pair = (a["block_id"], b["block_id"])
                    if pair not in existing:
                        pred = f"{_s(a.get('predicate'))}_{_s(b.get('predicate'))}"
                        self._edges.append((pair[0], pair[1], pred, 1.0))
                        existing.add(pair)
            return [{"num_links": len(self._edges)}]

        if "MATCH ()-[R:RELATED]->() RETURN COUNT(R)" in q_upper:
            return [{"n": len(self._edges)}]

        if "MATCH (A:FACT)-[R:RELATED]-(B:FACT)" in q_upper:
            bid = _s(p.get("bid"))
            excl_unv = "B.RELIABILITY <> 'UNVERIFIED'" in q_upper
            rows: list[dict[str, Any]] = []
            seen_neighbors: set[str] = set()
            for src, dst, _pred, _w in self._edges:
                other_id = dst if src == bid else (src if dst == bid else None)
                if not other_id or other_id in seen_neighbors:
                    continue
                other = self._facts.get(other_id)
                a_rec = self._facts.get(bid)
                if not other or not a_rec:
                    continue
                if excl_unv and _s(other.get("reliability")).upper() == "UNVERIFIED":
                    continue
                seen_neighbors.add(other_id)
                rows.append(_project({"a": a_rec, "b": other}, q))
            return rows

        if "MATCH (A:FACT)-[R:RELATED]->(B:FACT)" in q_upper:
            excl_unv = "UNVERIFIED" in q_upper
            same_pred = "LOWER(A.PREDICATE) = LOWER(B.PREDICATE)" in q_upper
            diff_end = "LOWER(A.SUBJECT) <> LOWER(B.OBJECT)" in q_upper
            rows = []
            for src, dst, _pred, _w in self._edges:
                a = self._facts.get(src)
                b = self._facts.get(dst)
                if not a or not b:
                    continue
                if excl_unv and (
                    _s(a.get("reliability")).upper() == "UNVERIFIED"
                    or _s(b.get("reliability")).upper() == "UNVERIFIED"
                ):
                    continue
                if same_pred and _s(a.get("predicate")).lower() != _s(b.get("predicate")).lower():
                    continue
                if diff_end and _s(a.get("subject")).lower() == _s(b.get("object")).lower():
                    continue
                rows.append(_project({"a": a, "b": b}, q))
            return rows

        if "MATCH (A:FACT), (B:FACT)" in q_upper:
            excl_unv = "UNVERIFIED" in q_upper
            diff_pred = "LOWER(A.PREDICATE) <> LOWER(B.PREDICATE)" in q_upper
            items = list(self._facts.values())
            rows = []
            for a in items:
                sa, oa = _s(a.get("subject")), _s(a.get("object"))
                if not sa or not oa:
                    continue
                if excl_unv and _s(a.get("reliability")).upper() == "UNVERIFIED":
                    continue
                for b in items:
                    if a["block_id"] >= b["block_id"]:
                        continue
                    sb, ob = _s(b.get("subject")), _s(b.get("object"))
                    if sa.lower() != sb.lower() or oa.lower() != ob.lower():
                        continue
                    if excl_unv and _s(b.get("reliability")).upper() == "UNVERIFIED":
                        continue
                    if (
                        diff_pred
                        and _s(a.get("predicate")).lower() == _s(b.get("predicate")).lower()
                    ):
                        continue
                    rows.append(_project({"a": a, "b": b}, q))
            return rows

        # Single-node MATCH (f:Fact) or MATCH (a:Fact)
        rows = []
        items = list(self._facts.values())
        if "ORDER BY F.QUALITY DESC" in q_upper:
            items.sort(key=lambda r: -int(r.get("quality") or 0))
        for rec in items:
            if (
                "CONTAINS LOWER($S)" in q_upper
                and _s(p.get("s")).lower() not in _s(rec.get("subject")).lower()
            ):
                continue
            if "F.RELIABILITY = $R" in q_upper and _s(rec.get("reliability")) != _s(p.get("r")):
                continue
            if "LOWER(A.PREDICATE) IN $PREDICATES" in q_upper:
                preds = {str(x).lower() for x in p.get("predicates", [])}
                if _s(rec.get("predicate")).lower() not in preds:
                    continue
            rows.append(_project({"f": rec, "a": rec}, q))
        limit_match = re.search(r"LIMIT\s+(\d+)\s*$", q_upper)
        if limit_match:
            rows = rows[: int(limit_match.group(1))]
        return rows
