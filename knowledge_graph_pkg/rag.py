"""Hybrid Vector + Cypher multi-hop Graph-RAG retriever."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .embeddings import cosine_similarity, get_embedder
from .graph_store_base import BaseGraphStore, _core_predicates_match, _s

_QUERY_STOPWORDS: Set[str] = {
    "is",
    "in",
    "of",
    "to",
    "on",
    "at",
    "by",
    "or",
    "an",
    "as",
    "be",
    "do",
    "if",
    "it",
    "no",
    "so",
    "up",
    "we",
    "he",
    "me",
    "my",
    "us",
    "the",
    "and",
    "for",
    "are",
    "was",
    "were",
    "has",
    "had",
    "not",
    "but",
    "what",
    "who",
    "how",
    "why",
    "where",
    "when",
}

_MAX_EMBEDDING_CACHE = 10000


def _extract_query_terms(query: str) -> List[str]:
    """Extract lowercase query terms of length >= 2, excluding common stopwords."""
    tokens = re.findall(r"[a-z0-9]+", _s(query).lower())
    return [w for w in tokens if len(w) >= 2 and w not in _QUERY_STOPWORDS]


def _term_matches(term: str, text_lower: str, text_tokens: Set[str]) -> bool:
    """Match short terms on token boundaries and longer terms via substring."""
    if len(term) <= 3:
        return term in text_tokens
    return term in text_lower


def _is_unverified(fact: Dict[str, Any]) -> bool:
    return _s(fact.get("reliability")).strip().upper() == "UNVERIFIED"


def _compute_pagerank_power(
    graph: Any, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6
) -> Dict[str, float]:
    """Pure-Python power-iteration PageRank fallback when scipy is not installed."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    x = {node: 1.0 / n for node in nodes}
    out_deg = {node: float(graph.out_degree(node)) for node in nodes}
    dangling = [node for node, deg in out_deg.items() if deg == 0.0]
    for _ in range(max_iter):
        xlast = x
        danglesum = alpha * sum(xlast[node] for node in dangling) / n
        x = {node: (1.0 - alpha) / n + danglesum for node in nodes}
        for u in nodes:
            deg = out_deg[u]
            if deg > 0.0:
                share = alpha * xlast[u] / deg
                for v in graph.successors(u):
                    x[v] += share
        if sum(abs(x[node] - xlast[node]) for node in nodes) < n * tol:
            break
    return x


class GraphRAGRetriever:
    """Hybrid Vector + Cypher Path Traversal retriever for Graph-RAG prompts."""

    def __init__(
        self,
        store: BaseGraphStore,
        embedder_type: Optional[str] = None,
        embedder_model: Optional[str] = None,
    ):
        self.store = store
        self.embedder = None
        if embedder_type:
            try:
                self.embedder = get_embedder(embedder_type, model=embedder_model)
            except Exception as exc:
                print(
                    f"[RAG] Warning: Could not load embedder ({exc}), falling back to keyword search."
                )
        self._cached_pagerank: Dict[str, float] = {}
        self._last_fact_count: int = -1
        self._last_edge_count: int = -1
        self._embedding_cache: Dict[str, List[float]] = {}
        self._vector_index: Any = None
        self._vector_index_sig: Optional[Tuple[Any, ...]] = None
        self._vector_fact_map: Dict[str, Dict[str, Any]] = {}

    def invalidate_cache(self) -> None:
        """Clear cached PageRank scores, statement embeddings, and vector index."""
        self._cached_pagerank = {}
        self._last_fact_count = -1
        self._last_edge_count = -1
        self._embedding_cache.clear()
        self._vector_index = None
        self._vector_index_sig = None
        self._vector_fact_map = {}

    def _embed_cached(self, text: str) -> Optional[List[float]]:
        if not self.embedder:
            return None
        key = _s(text)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        try:
            vec = self.embedder.embed_one(key)
        except Exception:
            return None
        if not isinstance(vec, list) or not vec:
            return None
        if len(self._embedding_cache) >= _MAX_EMBEDDING_CACHE:
            oldest = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest]
        self._embedding_cache[key] = vec
        return vec

    def retrieve_seeds(
        self, query: str, limit: int = 5, exclude_unverified: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve seed facts using semantic vector similarity or keyword matching."""
        if limit <= 0:
            return []

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

        if exclude_unverified:
            facts = [f for f in facts if not _is_unverified(f)]
            if not facts:
                return []

        if self.embedder:
            try:
                from .embeddings import VectorIndex

                sig = (
                    exclude_unverified,
                    tuple((_s(f.get("block_id")), _s(f.get("statement"))) for f in facts),
                )
                if self._vector_index is None or sig != self._vector_index_sig:
                    index = VectorIndex()
                    fact_map: Dict[str, Dict[str, Any]] = {}
                    for f in facts:
                        bid = _s(f.get("block_id"))
                        stmt = _s(f.get("statement"))
                        if not bid:
                            continue
                        stmt_vec = self._embed_cached(stmt)
                        if not stmt_vec:
                            continue
                        index.add_vector(bid, stmt_vec)
                        fact_map[bid] = f
                    self._vector_index = index
                    self._vector_index_sig = sig
                    self._vector_fact_map = fact_map

                query_vec = self._embed_cached(query)
                if query_vec and self._vector_index.vectors:
                    results = self._vector_index.query(query_vec, top_k=limit)
                    return [
                        self._vector_fact_map[bid]
                        for bid, _ in results
                        if bid in self._vector_fact_map
                    ]
            except Exception as exc:
                print(
                    f"[RAG] Embedding similarity calculation failed: {exc}. "
                    "Falling back to keyword match."
                )

        words = _extract_query_terms(query)
        if not words:
            return facts[:limit]

        scored_kw = []
        for f in facts:
            score = 0
            stmt_lower = _s(f.get("statement")).lower()
            subj_lower = _s(f.get("subject")).lower()
            obj_lower = _s(f.get("object")).lower()
            stmt_tokens = set(re.findall(r"[a-z0-9]+", stmt_lower))
            subj_tokens = set(re.findall(r"[a-z0-9]+", subj_lower))
            obj_tokens = set(re.findall(r"[a-z0-9]+", obj_lower))
            for w in words:
                if _term_matches(w, stmt_lower, stmt_tokens):
                    score += 1
                if _term_matches(w, subj_lower, subj_tokens):
                    score += 2
                if _term_matches(w, obj_lower, obj_tokens):
                    score += 2
            if score > 0:
                scored_kw.append((score, f))

        scored_kw.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_kw[:limit]]

    def calculate_pagerank(self) -> Dict[str, float]:
        """Compute PageRank scores over RELATED edges with node/edge cache invalidation."""
        import networkx as nx

        try:
            raw_count = self.store.count()
            current_count = (
                raw_count
                if isinstance(raw_count, int) and not isinstance(raw_count, bool)
                else -1
            )
        except Exception:
            current_count = -1

        current_edge_count = -1
        if hasattr(self.store, "edge_count"):
            try:
                raw_edges = self.store.edge_count()
                if isinstance(raw_edges, int) and not isinstance(raw_edges, bool):
                    current_edge_count = raw_edges
            except Exception:
                current_edge_count = -1

        if (
            current_count >= 0
            and current_count == self._last_fact_count
            and current_edge_count == self._last_edge_count
        ):
            return self._cached_pagerank

        try:
            facts = self.store.query("MATCH (f:Fact) RETURN f.block_id AS block_id")
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
        except ImportError:
            scores = _compute_pagerank_power(graph, alpha=0.85)
        except Exception:
            scores = {node: 1.0 / len(graph) for node in graph.nodes}

        self._cached_pagerank = scores
        self._last_fact_count = current_count
        self._last_edge_count = current_edge_count
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 2,
        pagerank_weight: float = 0.3,
        exclude_unverified: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid multi-hop retrieval over the knowledge graph."""
        if top_k <= 0:
            return []
        hops = max(0, int(hops))

        seeds = self.retrieve_seeds(
            query, limit=top_k, exclude_unverified=exclude_unverified
        )
        if not seeds:
            return []

        retrieved_ids = {s["block_id"] for s in seeds if s.get("block_id")}
        results = list(seeds)

        pr_scores = self.calculate_pagerank()
        max_pr = max(pr_scores.values()) if pr_scores else 1.0

        conflicted_pairs: Set[Tuple[str, str]] = set()
        try:
            conflicts = self.store.query(
                "MATCH (a:Fact), (b:Fact) "
                "WHERE a.subject <> '' AND a.object <> '' "
                "AND lower(a.subject) = lower(b.subject) AND lower(a.object) = lower(b.object) "
                "AND lower(a.predicate) <> lower(b.predicate) "
                "AND a.reliability <> 'UNVERIFIED' AND b.reliability <> 'UNVERIFIED' "
                "AND a.block_id < b.block_id "
                "RETURN a.subject AS subject, a.object AS object, "
                "a.predicate AS a_pred, b.predicate AS b_pred, "
                "a.reliability AS a_rel, b.reliability AS b_rel"
            )
            for c in conflicts:
                if (
                    _s(c.get("a_rel")).strip().upper() == "UNVERIFIED"
                    or _s(c.get("b_rel")).strip().upper() == "UNVERIFIED"
                ):
                    continue
                if "a_pred" in c or "b_pred" in c:
                    if not _core_predicates_match(
                        _s(c.get("a_pred")), _s(c.get("b_pred"))
                    ):
                        continue
                s = _s(c.get("subject")).lower().strip()
                o = _s(c.get("object")).lower().strip()
                if s and o:
                    conflicted_pairs.add((s, o))
        except Exception:
            pass

        current_level = list(seeds)
        for _ in range(hops):
            next_level = []
            for node in current_level:
                try:
                    where_clause = "WHERE a.block_id = $bid "
                    if exclude_unverified:
                        where_clause += "AND b.reliability <> 'UNVERIFIED' "

                    related = self.store.query(
                        "MATCH (a:Fact)-[r:RELATED]-(b:Fact) "
                        + where_clause
                        + "RETURN b.block_id AS block_id, b.statement AS statement, "
                        "b.subject AS subject, b.predicate AS predicate, b.object AS object, "
                        "b.reliability AS reliability, b.quality AS quality",
                        {"bid": node["block_id"]},
                    )
                    for r in related:
                        if exclude_unverified and _is_unverified(r):
                            continue
                        s = _s(r.get("subject")).lower().strip()
                        o = _s(r.get("object")).lower().strip()
                        if (s, o) in conflicted_pairs or (o, s) in conflicted_pairs:
                            continue

                        bid = r.get("block_id")
                        if bid and bid not in retrieved_ids:
                            retrieved_ids.add(bid)
                            results.append(r)
                            next_level.append(r)
                except Exception:
                    pass
            current_level = next_level

        query_vec = self._embed_cached(query) if self.embedder else None
        words = _extract_query_terms(query)
        scored_results = []

        for f in results:
            bid = f.get("block_id")
            pr_val = pr_scores.get(bid, 0.0)
            norm_pr = pr_val / max_pr if max_pr > 0 else 0.0

            sim = 0.0
            if self.embedder and query_vec:
                stmt_vec = self._embed_cached(_s(f.get("statement")))
                if stmt_vec:
                    try:
                        sim = cosine_similarity(query_vec, stmt_vec)
                    except Exception:
                        sim = 0.0
            elif words:
                combined_lower = " ".join(
                    [
                        _s(f.get("statement")),
                        _s(f.get("subject")),
                        _s(f.get("predicate")),
                        _s(f.get("object")),
                    ]
                ).lower()
                combined_tokens = set(re.findall(r"[a-z0-9]+", combined_lower))
                matches = sum(
                    1 for w in words if _term_matches(w, combined_lower, combined_tokens)
                )
                sim = min(1.0, matches / len(words))

            hybrid_score = (1.0 - pagerank_weight) * sim + pagerank_weight * norm_pr
            scored_results.append((hybrid_score, f))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_results[: top_k * 2]]

    def format_facts(self, facts: List[Dict[str, Any]]) -> str:
        """Format retrieved fact records as a markdown block for LLM prompts."""
        if not facts:
            return "No relevant facts found in the knowledge graph."

        lines = ["### Verified Knowledge Graph Facts:"]
        for f in facts:
            reliability_str = _s(f.get("reliability")).strip() or "UNVERIFIED"
            lines.append(
                f"- **{_s(f.get('subject'))}** {_s(f.get('predicate'))} **{_s(f.get('object'))}** "
                f"(Statement: \"{_s(f.get('statement'))}\", Reliability: {reliability_str})"
            )
        return "\n".join(lines)

    def format_context(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 2,
        exclude_unverified: bool = True,
    ) -> str:
        """Format the retrieved graph context as a markdown block for LLM prompts."""
        facts = self.retrieve(
            query, top_k=top_k, hops=hops, exclude_unverified=exclude_unverified
        )
        return self.format_facts(facts)
