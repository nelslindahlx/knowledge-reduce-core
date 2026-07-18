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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


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
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, ensure_ascii=False, indent=2)

    def _shard_path(self, drop: Drop) -> str:
        # Partition by year/month of creation for tidy growth.
        dt = drop.created_at[:7].replace("-", os.sep)  # YYYY/MM
        subdir = os.path.join(self.shards_dir, dt)
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{drop.drop_id}.jsonl")

    # ------------------------------------------------------------------ #
    def write_drop(self, drop: Drop) -> str:
        """Append a drop as a JSONL shard and record it in the manifest.

        Returns the shard file path. The shard's first line is a header
        record (the drop metadata + source_text); each subsequent line is
        one fact, so shards stream cleanly without loading everything.
        """
        path = self._shard_path(drop)
        with open(path, "w", encoding="utf-8") as fh:
            header = {k: _json_safe(v) for k, v in drop.to_dict().items() if k != "facts"}
            header["record"] = "header"
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for fact in drop.facts:
                safe = _json_safe(fact)
                fh.write(json.dumps({"record": "fact", **safe}, ensure_ascii=False) + "\n")

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
            shard = os.path.join(self.root, entry["shard"])
            if not os.path.isfile(shard):
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
            "sources": sorted({d.get("source") for d in drops}),
            "schema_version": self._manifest.get("schema_version", SCHEMA_VERSION),
        }
