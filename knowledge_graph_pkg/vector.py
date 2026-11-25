"""
Advanced query capabilities for knowledge graphs with vector embeddings.

This module provides vector-based semantic search, query expansion,
and advanced retrieval capabilities for knowledge graphs.
"""

import hashlib
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .core import KnowledgeGraph


def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class VectorKnowledgeGraph:
    """
    Class for vector-based knowledge graph operations.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, embedding_dim: int = 384):
        """
        Initialize a VectorKnowledgeGraph with a knowledge graph.
        """
        self.kg = knowledge_graph
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self._index = None
        self._use_external_model = False
        
    def generate_embeddings(self, use_external_model: bool = False) -> int:
        """
        Generate vector embeddings for all facts in the knowledge graph.
        """
        self._use_external_model = bool(use_external_model)
        self.embeddings = {}
        count = 0
        
        for node, data in self.kg.graph.nodes(data=True):
            if 'fact_statement' in data:
                if self._use_external_model:
                    embedding = self._generate_mock_embedding(data['fact_statement'])
                else:
                    embedding = self._hash_based_embedding(data['fact_statement'])
                    
                self.embeddings[node] = embedding
                count += 1
                
        self._build_index()
        return count
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform semantic search using vector similarity.
        """
        if top_k <= 0 or not self.embeddings:
            return []

        if self._use_external_model or self._is_external_model_available():
            query_embedding = self._generate_mock_embedding(query)
        else:
            query_embedding = self._hash_based_embedding(query)
            
        if self._index is not None:
            return self._search_with_index(query_embedding, top_k)
            
        similarities = []
        for fact_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((fact_id, similarity))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def find_similar_facts(self, fact_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find facts similar to a given fact using vector similarity.
        """
        if fact_id not in self.embeddings:
            raise ValueError(f"No embedding found for fact ID '{fact_id}'")
        if top_k <= 0:
            return []
            
        fact_embedding = self.embeddings[fact_id]
        
        if self._index is not None:
            results = self._search_with_index(fact_embedding, top_k + 1)
            return [(fid, score) for fid, score in results if fid != fact_id][:top_k]
            
        similarities = []
        for other_id, embedding in self.embeddings.items():
            if other_id != fact_id:
                similarity = self._cosine_similarity(fact_embedding, embedding)
                similarities.append((other_id, similarity))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def query_expansion(self, query: str, expansion_terms: int = 3) -> str:
        """
        Expand a query with semantically related terms.
        
        Args:
            query: Original query string
            expansion_terms: Number of terms to add to the query
            
        Returns:
            Expanded query string
        """
        # Simple query expansion using related terms
        # In a real implementation, this would use more sophisticated techniques
        
        # Extract main terms from the query
        terms = re.findall(r'\b\w+\b', query.lower())
        
        # Find related terms for each term
        expanded_terms = set(terms)
        
        for term in terms:
            related = self._find_related_terms(term, expansion_terms)
            expanded_terms.update(related)
            
        # Combine original query with expansion terms
        expanded_query = query + " " + " ".join(expanded_terms - set(terms))
        
        return expanded_query
        
    def cluster_facts(self, num_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster facts based on vector similarity.
        """
        if not self.embeddings:
            raise ValueError("No embeddings available. Generate embeddings first.")
        if num_clusters <= 0:
            raise ValueError("num_clusters must be a positive integer.")
            
        fact_ids = list(self.embeddings.keys())
        embeddings_matrix = np.array([self.embeddings[fid] for fid in fact_ids])
        k = min(int(num_clusters), len(fact_ids))
        
        rng = np.random.default_rng(42)
        indices = rng.choice(len(fact_ids), k, replace=False)
        centers = embeddings_matrix[indices]
        
        clusters = {i: [] for i in range(k)}
        
        for i, fact_id in enumerate(fact_ids):
            embedding = embeddings_matrix[i]
            distances = [np.linalg.norm(embedding - center) for center in centers]
            nearest_cluster = int(np.argmin(distances))
            clusters[nearest_cluster].append(fact_id)
            
        return clusters
        
    def save_embeddings(self, filename: str) -> None:
        """
        Save embeddings to a file atomically.
        """
        serializable_embeddings = {
            fact_id: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            for fact_id, embedding in self.embeddings.items()
        }
        
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{filename}.tmp.{os.getpid()}"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_embeddings, f)
        os.replace(tmp_path, filename)
            
    def load_embeddings(self, filename: str) -> int:
        """
        Load embeddings from a file.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            serialized_embeddings = json.load(f)
            
        self.embeddings = {
            fact_id: np.array(embedding) if isinstance(embedding, list) else embedding
            for fact_id, embedding in serialized_embeddings.items()
        }
        
        self._build_index()
        return len(self.embeddings)
        
    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic mock embedding using an isolated local RNG.
        """
        seed_val = _stable_hash_int(text) % (2**32)
        rng = np.random.default_rng(seed_val)
        embedding = rng.normal(0, 1, self.embedding_dim)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
        
    def _hash_based_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic hash-based embedding.
        """
        embedding = np.zeros(self.embedding_dim)
        words = re.findall(r'\b\w+\b', str(text or '').lower())
        
        for word in words:
            word_hash = _stable_hash_int(word) % self.embedding_dim
            embedding[word_hash % self.embedding_dim] += 1
            embedding[(word_hash + 1) % self.embedding_dim] += 0.5
            embedding[(word_hash + 2) % self.embedding_dim] += 0.25
            
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
        
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
        
    def _is_external_model_available(self) -> bool:
        """
        Check if an external embedding model is available.
        
        Returns:
            True if available, False otherwise
        """
        # In a real implementation, check for the presence of required libraries
        # For this demonstration, always return False
        return False
        
    def _find_related_terms(self, term: str, count: int) -> List[str]:
        """
        Find terms related to a given term.
        
        Args:
            term: Term to find related terms for
            count: Number of related terms to find
            
        Returns:
            List of related terms
        """
        # Simple implementation using word similarity
        # In a real implementation, use word embeddings or a thesaurus
        
        # Dictionary of some common related terms (very limited)
        related_terms = {
            'knowledge': ['information', 'data', 'wisdom', 'understanding'],
            'graph': ['network', 'chart', 'diagram', 'structure'],
            'data': ['information', 'facts', 'statistics', 'records'],
            'search': ['find', 'query', 'lookup', 'retrieve'],
            'semantic': ['meaning', 'linguistic', 'conceptual', 'contextual'],
            'vector': ['embedding', 'array', 'direction', 'magnitude'],
            'fact': ['information', 'truth', 'datum', 'statement'],
            'query': ['question', 'search', 'inquiry', 'request'],
            'similarity': ['resemblance', 'likeness', 'comparison', 'affinity'],
            'cluster': ['group', 'category', 'collection', 'classification']
        }
        
        if term in related_terms:
            return related_terms[term][:count]
        else:
            return []
            
    def _build_index(self) -> None:
        """
        Build an index for fast similarity search.
        
        In a real implementation, this would use a proper vector index like FAISS.
        This is a simplified placeholder implementation.
        """
        # For demonstration purposes, we're not implementing a real index
        # In a real implementation, use a library like FAISS or Annoy
        
        # Just set a flag indicating the index is "built"
        self._index = True
        
    def _search_with_index(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Search using the vector index.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of tuples containing (fact_id, similarity_score)
        """
        # Since we don't have a real index, fall back to linear search
        similarities = []
        
        for fact_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((fact_id, similarity))
            
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
