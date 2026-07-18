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
        self._cached_pagerank: Dict[str, float] = {}
        self._last_fact_count: int = -1

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
                from .embeddings import VectorIndex
                query_vec = self.embedder.embed_one(query)
                index = VectorIndex()
                fact_map = {}
                for f in facts:
                    bid = f["block_id"]
                    stmt = f.get("statement", "")
                    stmt_vec = self.embedder.embed_one(stmt)
                    index.add_vector(bid, stmt_vec)
                    fact_map[bid] = f
                
                results = index.query(query_vec, top_k=limit)
                return [fact_map[bid] for bid, _ in results]
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

    def calculate_pagerank(self) -> Dict[str, float]:
        """Load all facts and relationships from Kuzu DB into a NetworkX DiGraph
        and compute Page-Rank scores for all facts. Caches results in-memory."""
        import networkx as nx
        
        try:
            current_count = self.store.count()
            if not isinstance(current_count, (int, float)):
                current_count = -1
        except Exception:
            current_count = -1
            
        if current_count >= 0 and current_count == self._last_fact_count:
            return self._cached_pagerank

        try:
            # 1. Retrieve all facts (nodes)
            facts = self.store.query("MATCH (f:Fact) RETURN f.block_id AS block_id")
            # 2. Retrieve all relationships (edges)
            edges = self.store.query(
                "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
                "RETURN a.block_id AS src, b.block_id AS dst"
            )
        except Exception as exc:
            print(f"[RAG] Page-Rank calculation failed to query DB: {exc}")
            return {}

        graph = nx.DiGraph()
        for f in facts:
            if "block_id" in f:
                graph.add_node(f["block_id"])
        for e in edges:
            if isinstance(e, dict) and "src" in e and "dst" in e:
                graph.add_edge(e["src"], e["dst"])

        if len(graph) == 0:
            return {}

        try:
            scores = nx.pagerank(graph, alpha=0.85)
        except Exception:
            scores = {node: 1.0 / len(graph) for node in graph.nodes}
            
        self._cached_pagerank = scores
        self._last_fact_count = current_count
        return scores

    def retrieve(self, query: str, top_k: int = 5, hops: int = 2, pagerank_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Perform hybrid multi-hop retrieval:
        1. Retrieve seed facts.
        2. Traverse multi-hop paths to find adjacent facts.
        3. Score and rank all candidate facts using both semantic similarity and Page-Rank importance.
        """
        seeds = self.retrieve_seeds(query, limit=top_k)
        if not seeds:
            return []

        retrieved_ids = {s["block_id"] for s in seeds}
        results = list(seeds)

        # Compute Page-Rank scores for the network
        pr_scores = self.calculate_pagerank()

        # Multi-hop traversal (up to hops)
        current_level = list(seeds)
        for h in range(hops):
            next_level = []
            for node in current_level:
                try:
                    related = self.store.query(
                        "MATCH (a:Fact)-[r:RELATED]-(b:Fact) "
                        "WHERE a.block_id = $bid "
                        "RETURN b.block_id AS block_id, b.statement AS statement, "
                        "b.subject AS subject, b.predicate AS predicate, b.object AS object, "
                        "b.reliability AS reliability, b.quality AS quality",
                        {"bid": node["block_id"]}
                    )
                    for r in related:
                        if r["block_id"] not in retrieved_ids:
                            retrieved_ids.add(r["block_id"])
                            results.append(r)
                            next_level.append(r)
                except Exception:
                    pass
            current_level = next_level

        # Calculate final hybrid rankings
        scored_results = []
        query_vec = None
        if self.embedder:
            try:
                query_vec = self.embedder.embed_one(query)
            except Exception:
                pass

        for f in results:
            bid = f["block_id"]
            pr_val = pr_scores.get(bid, 0.0)

            # Max Page-Rank normalization to scale to [0, 1] range
            max_pr = max(pr_scores.values()) if pr_scores else 1.0
            norm_pr = pr_val / max_pr if max_pr > 0 else 0.0

            # Semantic similarity score
            sim = 0.0
            if self.embedder and query_vec:
                try:
                    stmt_vec = self.embedder.embed_one(f.get("statement", ""))
                    sim = cosine_similarity(query_vec, stmt_vec)
                except Exception:
                    pass
            else:
                # String matching fallback
                words = [w.lower() for w in query.split() if len(w) > 2]
                if words:
                    matches = sum(1 for w in words if w in f.get("statement", "").lower())
                    sim = min(1.0, matches / len(words))

            # Combine similarity and Page-Rank
            hybrid_score = (1.0 - pagerank_weight) * sim + pagerank_weight * norm_pr
            scored_results.append((hybrid_score, f))

        # Sort by hybrid score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_results[:top_k * 2]]

    def format_context(self, query: str, top_k: int = 5, hops: int = 2) -> str:
        """Format the retrieved graph context as a markdown block for LLM prompts."""
        facts = self.retrieve(query, top_k=top_k, hops=hops)
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
