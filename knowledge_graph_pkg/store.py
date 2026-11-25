"""
Knowledge-drop store (the write side of the knowledge factory).

A **drop** is one immutable shard produced by a single ingestion effort:
the facts extracted from one source, plus provenance (source URI, content
hash, timestamp) and lineage (extractor engine, filter settings, coref
flag, schema version). Drops are appended to the store as JSONL shards
under ``store/shards/YYYY/MM/`` and tracked in ``store/manifest.json``.

Design rules:

* **Immutable & append-only** -- a shard is written once; "edits" produce
  a new drop, never an in-place change. This keeps the corpus trustworthy
  for downstream training.
* **Self-describing** -- every drop carries the schema version and enough
  provenance/lineage to re-extract or audit it later.
* **Pure stdlib** -- JSONL + JSON, no database server.

Later sessions add a SQLite catalog (read-side index) and a ``compile``
step that assembles training sets as reproducible views over the store.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


def _safe_drop_id(drop_id: str) -> str:
    """Return a filesystem-safe single-segment identifier for a drop."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(drop_id or "")).strip("._")
    return cleaned or "drop"


def _partition_from_timestamp(created_at: Any) -> str:
    """Return a safe ``YYYY/MM`` relative partition path from ``created_at``."""
    prefix = str(created_at or "")[:7]
    if not re.fullmatch(r"\d{4}-\d{2}", prefix):
        prefix = _utc_now()[:7]
    year, month = prefix.split("-", 1)
    return os.path.join(year, month)


def _is_within(base: str, target: str) -> bool:
    """Return True iff ``target`` resolves inside directory ``base``."""
    base_abs = os.path.abspath(base)
    target_abs = os.path.abspath(target)
    try:
        return os.path.commonpath([base_abs, target_abs]) == base_abs
    except ValueError:
        return False


def _json_safe(value: Any) -> Any:
    """Convert a value into something JSON-serializable.

    Facts coming from the graph may carry an Enum (e.g. ReliabilityRating)
    or other non-primitive values; persist those as their name/string so
    shards stay plain JSON.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

# Bump when the drop schema changes in a breaking way. Stored on every drop
# and in the manifest so old shards remain interpretable.
SCHEMA_VERSION = 1


def content_hash(text: str) -> str:
    """Return the sha256 hex digest of ``text`` (used for source identity)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Drop:
    """One immutable knowledge shard from a single ingestion effort."""

    drop_id: str
    source: str
    source_hash: str
    facts: List[Dict[str, Any]]
    engine: str = "svo"
    filter_name: str = "standard"
    coref: bool = False
    created_at: str = field(default_factory=_utc_now)
    schema_version: int = SCHEMA_VERSION
    source_text: Optional[str] = None  # kept so we can re-extract later
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "drop_id": self.drop_id,
            "source": self.source,
            "source_hash": self.source_hash,
            "created_at": self.created_at,
            "engine": self.engine,
            "filter_name": self.filter_name,
            "coref": self.coref,
            "num_facts": len(self.facts),
            "facts": self.facts,
            "source_text": self.source_text,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Drop":
        return cls(
            drop_id=d["drop_id"],
            source=d["source"],
            source_hash=d["source_hash"],
            facts=d.get("facts", []),
            engine=d.get("engine", "svo"),
            filter_name=d.get("filter_name", "standard"),
            coref=d.get("coref", False),
            created_at=d.get("created_at", _utc_now()),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            source_text=d.get("source_text"),
            meta=d.get("meta", {}),
        )

    def summary(self) -> Dict[str, Any]:
        """The manifest-level record (no fact bodies)."""
        return {
            "drop_id": self.drop_id,
            "source": self.source,
            "source_hash": self.source_hash,
            "created_at": self.created_at,
            "engine": self.engine,
            "filter_name": self.filter_name,
            "coref": self.coref,
            "num_facts": len(self.facts),
            "schema_version": self.schema_version,
        }


class KnowledgeStore:
    """An append-only store of knowledge drops on disk."""

    def __init__(self, root: str):
        self.root = root
        self.shards_dir = os.path.join(root, "shards")
        self.manifest_path = os.path.join(root, "manifest.json")
        os.makedirs(self.shards_dir, exist_ok=True)
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------ #
    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"schema_version": SCHEMA_VERSION, "drops": []}

    def _save_manifest(self) -> None:
        tmp_path = f"{self.manifest_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.manifest_path)

    def _shard_path(self, drop: Drop) -> str:
        # Partition by year/month of creation for tidy growth.
        dt = _partition_from_timestamp(drop.created_at)
        subdir = os.path.join(self.shards_dir, dt)
        os.makedirs(subdir, exist_ok=True)
        safe_id = _safe_drop_id(drop.drop_id)
        candidate = os.path.join(subdir, f"{safe_id}.jsonl")
        version = 2
        while os.path.exists(candidate):
            candidate = os.path.join(subdir, f"{safe_id}-v{version}.jsonl")
            version += 1
        if not _is_within(self.shards_dir, candidate):
            raise ValueError(f"Unsafe shard path for drop_id {drop.drop_id!r}")
        return candidate

    # ------------------------------------------------------------------ #
    def write_drop(self, drop: Drop) -> str:
        """Append a drop as a JSONL shard and record it in the manifest.

        Returns the shard file path. The shard's first line is a header
        record (the drop metadata + source_text); each subsequent line is
        one fact, so shards stream cleanly without loading everything.
        """
        path = self._shard_path(drop)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            header = {k: _json_safe(v) for k, v in drop.to_dict().items() if k != "facts"}
            header["record"] = "header"
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for fact in drop.facts:
                safe = _json_safe(fact)
                fh.write(json.dumps({**safe, "record": "fact"}, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

        rel = os.path.relpath(path, self.root)
        entry = drop.summary()
        entry["shard"] = rel
        self._manifest["drops"].append(entry)
        self._save_manifest()
        return path

    def list_drops(self) -> List[Dict[str, Any]]:
        """Return manifest summaries for all drops (no fact bodies)."""
        return list(self._manifest.get("drops", []))

    def has_source_hash(self, source_hash: str) -> bool:
        """True if a drop with this source content hash already exists."""
        return any(d.get("source_hash") == source_hash
                   for d in self._manifest.get("drops", []))

    def iter_facts(self) -> Iterator[Dict[str, Any]]:
        """Yield every fact across all shards, tagged with its drop_id/source."""
        for entry in self._manifest.get("drops", []):
            rel_shard = entry.get("shard") or ""
            shard = os.path.join(self.root, rel_shard)
            if not rel_shard or not _is_within(self.root, shard) or not os.path.isfile(shard):
                continue
            with open(shard, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("record") == "fact":
                        rec = {k: v for k, v in rec.items() if k != "record"}
                        rec["_drop_id"] = entry["drop_id"]
                        rec["_source"] = entry["source"]
                        yield rec

    def stats(self) -> Dict[str, Any]:
        drops = self._manifest.get("drops", [])
        return {
            "total_drops": len(drops),
            "total_facts": sum(d.get("num_facts", 0) for d in drops),
            "sources": sorted({str(d["source"]) for d in drops if d.get("source") is not None}),
            "schema_version": self._manifest.get("schema_version", SCHEMA_VERSION),
        }

    def audit_summary(self) -> Dict[str, Any]:
        """Perform diagnostic analysis over the drops and facts in the store."""
        drops = self.list_drops()
        total_facts = 0
        reliability_counts: Dict[str, int] = {}
        missing_svo_count = 0
        duplicate_svo_count = 0

        seen_svo = set()

        for fact in self.iter_facts():
            total_facts += 1

            raw_rel = fact.get("reliability_rating") or fact.get("reliability") or "UNVERIFIED"
            if isinstance(raw_rel, Enum):
                rel = raw_rel.name
            else:
                rel = str(raw_rel).strip().upper() or "UNVERIFIED"
            reliability_counts[rel] = reliability_counts.get(rel, 0) + 1

            subj = str(fact.get("subject") or "").strip()
            pred = str(fact.get("predicate") or "").strip()
            obj = str(fact.get("object") or "").strip()
            stmt = str(fact.get("fact_statement") or fact.get("statement") or "").strip()

            if not subj or not pred or not obj or not stmt:
                missing_svo_count += 1

            svo_key = (subj.lower(), pred.lower(), obj.lower())
            if svo_key in seen_svo:
                duplicate_svo_count += 1
            else:
                seen_svo.add(svo_key)

        return {
            "total_drops": len(drops),
            "total_facts": total_facts,
            "reliability_tier_distribution": reliability_counts,
            "missing_fields_count": missing_svo_count,
            "duplicate_svo_count": duplicate_svo_count,
        }

