"""Recursive model-weight crawler with logprob entropy filtering."""

from enum import Enum
import math
from typing import Any, Dict, List, Set, Tuple

from .core import KnowledgeGraph
from .extractor_base import get_extractor
from .semantic import SemanticKnowledgeGraph


class ModelCrawler:
    """Recursively crawls a model backend's weights to harvest knowledge facts."""

    def __init__(self, backend: Any):
        self.backend = backend
        self.model_name = getattr(backend, "model", "unknown")

    def crawl(
        self,
        seed_topic: str,
        max_depth: int = 2,
        concepts_per_level: int = 3,
        logprob_threshold: float = -1.5,
        engine: str = "svo",
        max_topics: int = 100,
    ) -> List[Dict[str, Any]]:
        """Crawl the model recursively starting from a seed topic."""
        seed_clean = str(seed_topic or "").strip()
        if not seed_clean:
            return []

        queue: List[Tuple[str, int]] = [(seed_clean, 0)]
        queued_lower: Set[str] = {seed_clean.lower()}
        visited: Set[str] = set()
        all_extracted_facts: List[Dict[str, Any]] = []

        extractor = get_extractor(engine) if engine != "svo" else None

        print(
            f"[Crawler] Starting crawl for seed '{seed_clean}' (Max Depth: {max_depth})..."
        )

        topics_crawled = 0
        while queue and topics_crawled < max(0, int(max_topics)):
            topic, depth = queue.pop(0)
            topic_clean = topic.strip().lower()
            if not topic_clean or topic_clean in visited or depth > max_depth:
                continue

            visited.add(topic_clean)
            topics_crawled += 1
            print(f"[Crawler] Depth {depth}: Crawling topic '{topic}'...")

            prompt = f"Provide a concise list of factual assertions about '{topic}'."
            raw_text, avg_logprob = self.backend.generate_text_with_logprobs(prompt)
            text = "" if raw_text is None else str(raw_text)
            if not text.strip():
                continue

            avg_lp = float(avg_logprob or 0.0)
            if avg_lp < logprob_threshold and avg_lp != 0.0:
                print(
                    f"[Crawler] Skipping topic '{topic}' due to high uncertainty "
                    f"(avg logprob: {avg_lp:.3f})"
                )
                continue

            kg = KnowledgeGraph()
            skg = SemanticKnowledgeGraph(kg)
            skg.create_facts_from_text(
                text,
                source_id=f"crawler-{topic_clean}",
                extractor=extractor,
            )

            extracted: List[Dict[str, Any]] = []
            for node_id, data in kg.graph.nodes.items():
                raw_rel = data.get("reliability_rating", "UNVERIFIED")
                if isinstance(raw_rel, Enum):
                    rel_str = raw_rel.name
                else:
                    rel_str = str(raw_rel or "UNVERIFIED").strip().upper()
                extracted.append(
                    {
                        "fact_id": node_id,
                        "fact_statement": data.get("fact_statement"),
                        "subject": data.get("subject"),
                        "predicate": data.get("predicate"),
                        "object": data.get("object"),
                        "question": data.get("question"),
                        "answer": data.get("answer"),
                        "category": data.get("category"),
                        "tags": data.get("tags"),
                        "reliability_rating": rel_str,
                    }
                )

            conf_pct = math.exp(min(0.0, avg_lp))
            print(
                f"[Crawler] Extracted {len(extracted)} facts for topic '{topic}' "
                f"(Confidence: {conf_pct:.1%})"
            )

            new_concepts: List[str] = []
            seen_new: Set[str] = set()
            for f in extracted:
                f["source_models"] = [self.model_name]
                f["avg_logprob"] = avg_lp
                f["model_provenance"] = {
                    "model": self.model_name,
                    "source": f"crawler-{topic_clean}",
                }
                all_extracted_facts.append(f)

                for key in ("subject", "object"):
                    val = str(f.get(key) or "").strip()
                    low = val.lower()
                    if (
                        val
                        and low not in visited
                        and low not in queued_lower
                        and low not in seen_new
                    ):
                        new_concepts.append(val)
                        seen_new.add(low)

            for c in new_concepts[: max(0, int(concepts_per_level))]:
                queue.append((c, depth + 1))
                queued_lower.add(c.lower())

        return all_extracted_facts
