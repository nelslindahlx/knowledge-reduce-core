"""
Advanced query capabilities for knowledge graphs with vector embeddings.

This module provides vector-based semantic search, query expansion,
and advanced retrieval capabilities for knowledge graphs.
"""

import numpy as np
import re
import json
from typing import Dict, List, Any, Union, Optional, Tuple, Set, Callable
from datetime import datetime
from .core import KnowledgeGraph, ReliabilityRating

class VectorKnowledgeGraph:
    """
    Class for vector-based knowledge graph operations.
    
    This class enhances knowledge graphs with vector embeddings for
    semantic search, similarity matching, and advanced query capabilities.
    
    Attributes:
        kg: The knowledge graph to enhance with vector capabilities
        embedding_dim: Dimension of vector embeddings
        embeddings: Dictionary mapping fact IDs to vector embeddings
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, embedding_dim: int = 384):
        """
        Initialize a VectorKnowledgeGraph with a knowledge graph.
        
        Args:
            knowledge_graph: KnowledgeGraph instance to enhance
            embedding_dim: Dimension of vector embeddings
        """
        self.kg = knowledge_graph
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self._index = None
        
    def generate_embeddings(self, use_external_model: bool = False) -> int:
        """
        Generate vector embeddings for all facts in the knowledge graph.
        
        Args:
            use_external_model: Whether to use an external embedding model
                               (if False, uses a simpler hash-based approach)
                               
        Returns:
            Number of embeddings generated
        """
        count = 0
        
        for node, data in self.kg.graph.nodes(data=True):
            if 'fact_statement' in data:
                if use_external_model:
                    # In a real implementation, this would use a proper embedding model
                    # like sentence-transformers, OpenAI embeddings, etc.
                    # This is a simplified placeholder implementation
                    embedding = self._generate_mock_embedding(data['fact_statement'])
                else:
                    # Simple hash-based embedding for demonstration
                    embedding = self._hash_based_embedding(data['fact_statement'])
                    
                self.embeddings[node] = embedding
                count += 1
                
        # Build the index for fast similarity search
        self._build_index()
                
        return count
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples containing (fact_id, similarity_score)
        """
        # Generate query embedding
        if self._is_external_model_available():
            query_embedding = self._generate_mock_embedding(query)
        else:
            query_embedding = self._hash_based_embedding(query)
            
        # If index is available, use it for fast search
        if self._index is not None:
            return self._search_with_index(query_embedding, top_k)
            
        # Otherwise, perform linear search
        similarities = []
        
        for fact_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((fact_id, similarity))
            
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
        
    def find_similar_facts(self, fact_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find facts similar to a given fact using vector similarity.
        
        Args:
            fact_id: ID of the fact to find similar facts for
            top_k: Number of top results to return
            
        Returns:
            List of tuples containing (fact_id, similarity_score)
        """
        if fact_id not in self.embeddings:
            raise ValueError(f"No embedding found for fact ID '{fact_id}'")
            
        fact_embedding = self.embeddings[fact_id]
        
        # If index is available, use it for fast search
        if self._index is not None:
            results = self._search_with_index(fact_embedding, top_k + 1)  # +1 to account for self-match
            # Filter out the query fact itself
            return [(id, score) for id, score in results if id != fact_id]
            
        # Otherwise, perform linear search
        similarities = []
        
        for other_id, embedding in self.embeddings.items():
            if other_id != fact_id:  # Skip the query fact
                similarity = self._cosine_similarity(fact_embedding, embedding)
                similarities.append((other_id, similarity))
                
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
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
        
        Args:
            num_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster IDs to lists of fact IDs
        """
        if not self.embeddings:
            raise ValueError("No embeddings available. Generate embeddings first.")
            
        # Simple k-means clustering implementation
        # In a real implementation, use a proper clustering library
        
        # Convert embeddings to a matrix
        fact_ids = list(self.embeddings.keys())
        embeddings_matrix = np.array([self.embeddings[fid] for fid in fact_ids])
        
        # Initialize cluster centers randomly
        indices = np.random.choice(len(fact_ids), num_clusters, replace=False)
        centers = embeddings_matrix[indices]
        
        # Perform k-means clustering (simplified)
        clusters = {i: [] for i in range(num_clusters)}
        
        # Assign facts to nearest cluster
        for i, fact_id in enumerate(fact_ids):
            embedding = embeddings_matrix[i]
            
            # Find nearest center
            distances = [np.linalg.norm(embedding - center) for center in centers]
            nearest_cluster = np.argmin(distances)
            
            # Assign to cluster
            clusters[nearest_cluster].append(fact_id)
            
        return clusters
        
    def save_embeddings(self, filename: str) -> None:
        """
        Save embeddings to a file.
        
        Args:
            filename: Path to save the embeddings
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = {
            fact_id: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            for fact_id, embedding in self.embeddings.items()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_embeddings, f)
            
    def load_embeddings(self, filename: str) -> int:
        """
        Load embeddings from a file.
        
        Args:
            filename: Path to load the embeddings from
            
        Returns:
            Number of embeddings loaded
        """
        with open(filename, 'r', encoding='utf-8') as f:
            serialized_embeddings = json.load(f)
            
        # Convert lists back to numpy arrays
        self.embeddings = {
            fact_id: np.array(embedding) if isinstance(embedding, list) else embedding
            for fact_id, embedding in serialized_embeddings.items()
        }
        
        # Rebuild the index
        self._build_index()
        
        return len(self.embeddings)
        
    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """
        Generate a mock embedding for demonstration purposes.
        
        In a real implementation, this would use a proper embedding model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Vector embedding as numpy array
        """
        # Create a deterministic but somewhat meaningful embedding based on text
        # This is just for demonstration - not for real use
        
        # Hash the text to get a seed
        hash_val = hash(text) % (2**32)
        np.random.seed(hash_val)
        
        # Generate a random embedding
        embedding = np.random.normal(0, 1, self.embedding_dim)
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
        
    def _hash_based_embedding(self, text: str) -> np.ndarray:
        """
        Generate a simple hash-based embedding.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Vector embedding as numpy array
        """
        # Initialize embedding vector
        embedding = np.zeros(self.embedding_dim)
        
        # Normalize and tokenize text
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # For each word, update specific dimensions based on hash
        for word in words:
            word_hash = hash(word) % self.embedding_dim
            embedding[word_hash % self.embedding_dim] += 1
            embedding[(word_hash + 1) % self.embedding_dim] += 0.5
            embedding[(word_hash + 2) % self.embedding_dim] += 0.25
            
        # Normalize to unit length if not zero
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
