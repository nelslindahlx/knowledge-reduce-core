"""
Semantic embeddings and NumPy vector search utilities (ModelReduce Session 2).
"""

from collections import OrderedDict
import math
import threading
from typing import Any, Callable, Dict, List, Optional


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two equal-length vectors, in ``[-1, 1]``."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class EmbeddingCache:
    """Thread-safe, bounded LRU memoizer for embedding functions."""

    def __init__(self, embed_one: Callable[[str], List[float]], max_size: int = 10000):
        self._embed_one = embed_one
        self._max_size = max(0, int(max_size))
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, text: str) -> List[float]:
        with self._lock:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                return cached

        vec = self._embed_one(text)
        if not isinstance(vec, list) or len(vec) == 0:
            return list(vec) if isinstance(vec, list) else []

        with self._lock:
            if self._max_size > 0:
                if text in self._cache:
                    self._cache.move_to_end(text)
                else:
                    if len(self._cache) >= self._max_size:
                        self._cache.popitem(last=False)
                    self._cache[text] = vec
        return vec


class OllamaEmbedder:
    """Semantic embeddings from a local Ollama embedding model."""

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise ImportError(
                "OllamaEmbedder requires the model-reduce extra: "
                "pip install knowledgereduce[model-reduce]"
            ) from exc
        self._ollama = ollama
        try:
            self.client = ollama.Client(host=host, timeout=timeout)
        except TypeError:
            self.client = ollama.Client(host=host)
        self.model = model
        self._cache = EmbeddingCache(self._embed_one_uncached)

    def _embed_one_uncached(self, text: str) -> List[float]:
        resp = self.client.embeddings(model=self.model, prompt=text)
        if isinstance(resp, dict):
            return list(resp.get("embedding") or [])
        return list(getattr(resp, "embedding", []) or [])

    def embed_one(self, text: str) -> List[float]:
        """Return the (cached) embedding vector for ``text``."""
        return self._cache.get(text)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (cached per text)."""
        return [self.embed_one(t) for t in texts]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between the embeddings of two texts."""
        return cosine_similarity(self.embed_one(a), self.embed_one(b))


class SentenceTransformersEmbedder:
    """Offline semantic embeddings from a local sentence-transformers model."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformersEmbedder requires the sentence-transformers extra: "
                "pip install knowledgereduce[sentence-transformers]"
            ) from exc
        self.model = model
        self.client = SentenceTransformer(model)
        self._cache = EmbeddingCache(self._embed_one_uncached)

    def _embed_one_uncached(self, text: str) -> List[float]:
        vector = self.client.encode(text, convert_to_numpy=True)
        return [float(x) for x in vector]

    def embed_one(self, text: str) -> List[float]:
        """Return the (cached) embedding vector for ``text``."""
        return self._cache.get(text)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (cached per text)."""
        return [self.embed_one(t) for t in texts]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between the embeddings of two texts."""
        return cosine_similarity(self.embed_one(a), self.embed_one(b))


LocalEmbedder = OllamaEmbedder


def get_embedder(embedder_type: str, model: Optional[str] = None, **kwargs) -> Any:
    """Factory function to instantiate the correct embeddings backend."""
    kind = str(embedder_type or "").lower()
    if kind == "ollama":
        return OllamaEmbedder(
            model=model or "mxbai-embed-large",
            host=kwargs.get("host", "http://localhost:11434"),
            timeout=kwargs.get("timeout", 60.0),
        )
    elif kind == "sentence-transformers":
        return SentenceTransformersEmbedder(model=model or "all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


class VectorIndex:
    """NumPy-vectorized similarity search index for fast cosine matching."""

    def __init__(self):
        import numpy as np

        self._np = np
        self.ids: List[str] = []
        self.vectors: List[List[float]] = []
        self._id_to_idx: Dict[str, int] = {}
        self._dim: Optional[int] = None
        self._matrix = None
        self._norms = None

    def add_vector(self, vector_id: str, vector: List[float]) -> None:
        """Add or update a non-empty fixed-dimension vector in the index."""
        if not isinstance(vector, (list, tuple)) or len(vector) == 0:
            raise ValueError("VectorIndex requires a non-empty vector.")
        vec = [float(x) for x in vector]
        if vector_id in self._id_to_idx:
            idx = self._id_to_idx[vector_id]
            if self._dim is not None and len(vec) != self._dim and len(self.vectors) > 1:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self._dim}, got {len(vec)}."
                )
            self._dim = len(vec)
            self.vectors[idx] = vec
        else:
            if self._dim is None:
                self._dim = len(vec)
            elif len(vec) != self._dim:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self._dim}, got {len(vec)}."
                )
            self._id_to_idx[vector_id] = len(self.ids)
            self.ids.append(vector_id)
            self.vectors.append(vec)
        self._matrix = None
        self._norms = None

    def remove_vector(self, vector_id: str) -> None:
        """Remove a vector from the index by ID."""
        idx = self._id_to_idx.pop(vector_id, None)
        if idx is not None:
            self.ids.pop(idx)
            self.vectors.pop(idx)
            for i in range(idx, len(self.ids)):
                self._id_to_idx[self.ids[i]] = i
            if not self.vectors:
                self._dim = None
            self._matrix = None
            self._norms = None

    def _build(self) -> None:
        """Construct/cache the numpy matrix and precompute L2 norms."""
        if not self.vectors:
            self._matrix = self._np.zeros((0, 0))
            self._norms = self._np.zeros((0,))
            return
        self._matrix = self._np.array(self.vectors, dtype=self._np.float32)
        self._norms = self._np.linalg.norm(self._matrix, axis=1)
        self._norms[self._norms == 0.0] = 1.0

    def query(self, query_vector: List[float], top_k: int = 5) -> List[tuple]:
        """Search the index for the most similar vectors."""
        if not self.vectors or top_k <= 0:
            return []
        if not query_vector:
            return []
        if self._dim is not None and len(query_vector) != self._dim:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self._dim}, got {len(query_vector)}."
            )

        if self._matrix is None:
            self._build()

        np = self._np
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return [(vid, 0.0) for vid in self.ids[:top_k]]

        dot_products = np.dot(self._matrix, q)
        similarities = dot_products / (self._norms * q_norm)
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices[:top_k]:
            results.append((self.ids[idx], float(similarities[idx])))
        return results
