import os
import math
from typing import Any, Dict, List, Set, Tuple
from .model_probe import get_backend
from .core import KnowledgeGraph
from .semantic import SemanticKnowledgeGraph
from .extractor_base import get_extractor

class ModelCrawler:
    """Recursively crawls a model backend's weights to harvest knowledge facts."""

    def __init__(self, backend: Any):
        self.backend = backend
        self.model_name = getattr(backend, "model", "unknown")

    def crawl(self, seed_topic: str, max_depth: int = 2,
              concepts_per_level: int = 3,
              logprob_threshold: float = -1.5,
              engine: str = "svo") -> List[Dict[str, Any]]:
        """Crawl the model recursively starting from a seed topic."""
        queue: List[Tuple[str, int]] = [(seed_topic, 0)]
        visited: Set[str] = set()
        all_extracted_facts: List[Dict[str, Any]] = []

        extractor = None
        if engine != "svo":
            try:
                extractor = get_extractor(engine)
            except ImportError:
                pass

        print(f"[Crawler] Starting crawl for seed '{seed_topic}' (Max Depth: {max_depth})...")

        while queue:
            topic, depth = queue.pop(0)
            topic_clean = topic.strip().lower()
            if not topic_clean or topic_clean in visited or depth > max_depth:
                continue

            visited.add(topic_clean)
            print(f"[Crawler] Depth {depth}: Crawling topic '{topic}'...")

            # 1. Query model backend for explanation
            prompt = f"Provide a concise list of factual assertions about '{topic}'."
            text, avg_logprob = self.backend.generate_text_with_logprobs(prompt)
            if not text.strip():
                continue

            # 2. Entropy Profiling: check logprob thresholds
            if avg_logprob < logprob_threshold and avg_logprob != 0.0:
                print(f"[Crawler] Skipping topic '{topic}' due to high uncertainty (avg logprob: {avg_logprob:.3f})")
                continue

            # 3. Extract facts from text
            kg = KnowledgeGraph()
            skg = SemanticKnowledgeGraph(kg)
            skg.create_facts_from_text(
                text,
                source_id=f"crawler-{topic_clean}",
                extractor=extractor
            )

            extracted = [
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
                    "reliability_rating": data.get("reliability_rating", "unverified")
                }
                for node_id, data in kg.graph.nodes.items()
            ]
            print(f"[Crawler] Extracted {len(extracted)} facts for topic '{topic}' (Confidence: {math.exp(avg_logprob):.1%})")

            # 4. Save facts and queue follow-up topics
            new_concepts = []
            for f in extracted:
                f["source_models"] = [self.model_name]
                f["avg_logprob"] = avg_logprob
                all_extracted_facts.append(f)

                subj = f.get("subject", "")
                obj = f.get("object", "")
                if subj and subj.lower() not in visited and subj.lower() not in [q[0].lower() for q in queue]:
                    new_concepts.append(subj)
                if obj and obj.lower() not in visited and obj.lower() not in [q[0].lower() for q in queue]:
                    new_concepts.append(obj)

            # Limit queue additions per level
            for c in new_concepts[:concepts_per_level]:
                queue.append((c, depth + 1))

        return all_extracted_facts
