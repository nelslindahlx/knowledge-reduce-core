"""
Persistent fact store for cross-run deduplication.

When you distill many documents over time you don't want the same fact
re-emitted on every run. :class:`FactStore` keeps a set of normalized
fact fingerprints, optionally persisted to a JSON file, so repeated runs
only surface genuinely new facts.

Normalization is lowercase + whitespace-collapsed, so trivial spacing or
casing differences are treated as the same fact.
"""

import json
import os
import re
from typing import Optional, Set


def _normalize(statement: str) -> str:
    return re.sub(r"\s+", " ", statement.strip().lower())


class FactStore:
    """A set of seen fact fingerprints with optional JSON persistence."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._seen: Set[str] = set()

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, statement: str) -> bool:
        return _normalize(statement) in self._seen

    def add(self, statement: str) -> bool:
        """Record a fact. Returns True if new, False if already seen."""
        key = _normalize(statement)
        if not key or key in self._seen:
            return False
        self._seen.add(key)
        return True

    def load(self) -> "FactStore":
        """Load fingerprints from ``path`` if it exists. Returns self."""
        if self.path and os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._seen.update(data.get("seen", []))
        return self

    def save(self) -> None:
        """Persist fingerprints to ``path`` (no-op if path is unset)."""
        if not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"seen": sorted(self._seen)}, fh, ensure_ascii=False, indent=0)
