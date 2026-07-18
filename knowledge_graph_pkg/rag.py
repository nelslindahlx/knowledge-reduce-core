import os
from typing import Any, Dict, List, Optional
from .kuzu_store import KuzuStore
from .embeddings import get_embedder, cosine_similarity

class GraphRAGRetriever:
    """Hybrid Vector + Cypher Path Traversal retriever for Graph-RAG prompts."""

    def __init__(self, store: KuzuStore, embedder_type: Optional[str] = None,
                 embedder_model: Optional[str] = None):
        self.store = store
        self.embedder = None
        if embedder_type:
            try:
                self.embedder = get_embedder(embedder_type, model=embedder_model)
            except Exception as exc:
                print(f"[RAG] Warning: Could not load embedder ({exc}), falling back to keyword search.")

    def retrieve_seeds(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve seed facts using semantic vector similarity or keyword matching."""
        try:
            facts = self.store.query(
                "MATCH (f:Fact) "
                "RETURN f.block_id AS block_id, f.statement AS statement, "
                "f.subject AS subject, f.predicate AS predicate, f.object AS object, "
                "f.reliability AS reliability, f.quality AS quality"
            )
        except Exception:
            return []

        if not facts:
            return []

        # If embedder is active, rank by cosine similarity
        if self.embedder:
            try:
                query_vec = self.embedder.embed_one(query)
                scored = []
                for f in facts:
                    stmt = f.get("statement", "")
                    stmt_vec = self.embedder.embed_one(stmt)
                    sim = cosine_similarity(query_vec, stmt_vec)
                    scored.append((sim, f))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [item[1] for item in scored[:limit]]
            except Exception as exc:
                print(f"[RAG] Embedding similarity calculation failed: {exc}. Falling back to keyword match.")

        # Fallback keyword match: contains search on statement, subject or object
        words = [w.lower() for w in query.split() if len(w) > 2]
        if not words:
            return facts[:limit]

        scored_kw = []
        for f in facts:
            score = 0
            stmt_lower = f.get("statement", "").lower()
            subj_lower = f.get("subject", "").lower()
            obj_lower = f.get("object", "").lower()
            for w in words:
                if w in stmt_lower:
                    score += 1
                if w in subj_lower:
                    score += 2
                if w in obj_lower:
                    score += 2
            if score > 0:
                scored_kw.append((score, f))

        scored_kw.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_kw[:limit]]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval: get seed facts, then traverse their connected graph paths."""
        seeds = self.retrieve_seeds(query, limit=top_k)
        if not seeds:
            return []

        retrieved_ids = {s["block_id"] for s in seeds}
        results = list(seeds)

        # For each seed, traverse adjacent related facts in Kuzu DB
        for s in seeds:
            try:
                related = self.store.query(
                    "MATCH (a:Fact)-[r:RELATED]-(b:Fact) "
                    "WHERE a.block_id = $bid "
                    "RETURN b.block_id AS block_id, b.statement AS statement, "
                    "b.subject AS subject, b.predicate AS predicate, b.object AS object, "
                    "b.reliability AS reliability, b.quality AS quality, "
                    "r.predicate AS rel_predicate",
                    {"bid": s["block_id"]}
                )
                for r in related:
                    if r["block_id"] not in retrieved_ids:
                        retrieved_ids.add(r["block_id"])
                        results.append(r)
            except Exception:
                pass

        return results

    def format_context(self, query: str, top_k: int = 5) -> str:
        """Format the retrieved graph context as a markdown block for LLM prompts."""
        facts = self.retrieve(query, top_k=top_k)
        if not facts:
            return "No relevant facts found in the knowledge graph."

        lines = ["### Verified Knowledge Graph Facts:"]
        for f in facts:
            reliability_str = f.get("reliability", "UNVERIFIED")
            lines.append(
                f"- **{f['subject']}** {f['predicate']} **{f['object']}** "
                f"(Statement: \"{f['statement']}\", Reliability: {reliability_str})"
            )
        return "\n".join(lines)
