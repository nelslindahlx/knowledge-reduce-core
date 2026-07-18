"""
SQLite catalog / index (the read-side index of the knowledge factory).

The store keeps facts in JSONL shards (good for append-only durability,
bad for ad-hoc queries). :class:`Catalog` builds a SQLite index over all
stored facts so they are findable later -- by source, reliability,
category, or quality -- without scanning every shard.

The catalog is a derived artifact: it can always be rebuilt from the
store, so it is safe to delete and regenerate. Pure stdlib (``sqlite3``).
"""

import sqlite3
from typing import Any, Dict, List, Optional


class Catalog:
    """A rebuildable SQLite index over a :class:`KnowledgeStore`."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drop_id TEXT,
                source TEXT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                statement TEXT,
                question TEXT,
                answer TEXT,
                reliability TEXT,
                quality INTEGER,
                category TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_source ON facts(source);
            CREATE INDEX IF NOT EXISTS idx_quality ON facts(quality);
            CREATE INDEX IF NOT EXISTS idx_reliability ON facts(reliability);
            CREATE INDEX IF NOT EXISTS idx_category ON facts(category);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    def rebuild(self, store) -> int:
        """Drop and repopulate the index from ``store``. Returns row count."""
        self._conn.execute("DELETE FROM facts")
        rows = []
        for f in store.iter_facts():
            rows.append((
                f.get("_drop_id"),
                f.get("_source"),
                f.get("subject"),
                f.get("predicate"),
                f.get("object"),
                f.get("fact_statement"),
                f.get("question"),
                f.get("answer"),
                f.get("reliability_rating"),
                int(f.get("quality_score") or 0),
                f.get("category"),
            ))
        self._conn.executemany(
            "INSERT INTO facts (drop_id, source, subject, predicate, object, "
            "statement, question, answer, reliability, quality, category) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def query(self, source: Optional[str] = None, reliability: Optional[str] = None,
              category: Optional[str] = None, min_quality: Optional[int] = None,
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query indexed facts with optional filters; returns row dicts."""
        clauses, params = [], []
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if reliability is not None:
            clauses.append("reliability = ?")
            params.append(reliability)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if min_quality is not None:
            clauses.append("quality >= ?")
            params.append(min_quality)

        sql = "SELECT * FROM facts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY quality DESC, id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def stats(self) -> Dict[str, Any]:
        """Return store-wide summary statistics from the index."""
        total = self.count()
        drops = self._conn.execute("SELECT COUNT(DISTINCT drop_id) FROM facts").fetchone()[0]
        sources = self._conn.execute("SELECT COUNT(DISTINCT source) FROM facts").fetchone()[0]
        by_rel = {
            row["reliability"]: row["n"]
            for row in self._conn.execute(
                "SELECT reliability, COUNT(*) AS n FROM facts GROUP BY reliability"
            ).fetchall()
        }
        return {
            "total_facts": total,
            "total_drops": drops,
            "sources": sources,
            "by_reliability": by_rel,
        }

    def close(self) -> None:
        self._conn.close()
