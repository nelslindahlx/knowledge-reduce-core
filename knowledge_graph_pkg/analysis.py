"""
Advanced knowledge graph operations and analysis.

This module provides advanced functionality for knowledge graph operations,
including semantic analysis, entity extraction, and advanced querying capabilities.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Set
from .core import KnowledgeGraph, ReliabilityRating


class KnowledgeGraphAnalyzer:
    """
    Class for analyzing knowledge graphs and extracting insights.
    
    This class provides methods for analyzing the structure and content of
    knowledge graphs, identifying important nodes, and extracting patterns.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Initialize a KnowledgeGraphAnalyzer with a knowledge graph.
        
        Args:
            knowledge_graph: KnowledgeGraph instance to analyze
        """
        self.kg = knowledge_graph
        
    def get_central_facts(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify the most central facts in the knowledge graph using betweenness centrality.
        
        Args:
            top_n: Number of top central facts to return
            
        Returns:
            List of tuples containing (fact_id, centrality_score)
        """
        if len(self.kg.graph) == 0:
            return []
            
        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(self.kg.graph)
        
        # Sort facts by centrality score
        sorted_facts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N facts
        return sorted_facts[:top_n]
        
    def get_fact_communities(self) -> Dict[str, int]:
        """
        Identify communities of related facts using the Louvain method.
        
        Returns:
            Dictionary mapping fact_id to community number
        """
        if len(self.kg.graph) == 0:
            return {}
            
        # Convert directed graph to undirected for community detection
        undirected_graph = self.kg.graph.to_undirected()
        
        # Use NetworkX's community detection
        try:
            from community import best_partition
            return best_partition(undirected_graph)
        except ImportError:
            # Fallback to connected components if python-louvain is not installed
            communities = {}
            for i, component in enumerate(nx.connected_components(undirected_graph)):
                for node in component:
                    communities[node] = i
            return communities
            
    def get_reliability_summary(self) -> Dict[str, int]:
        """
        Get a summary of fact reliability ratings in the knowledge graph.
        
        Returns:
            Dictionary mapping reliability rating names to counts
        """
        summary = {rating.name: 0 for rating in ReliabilityRating}
        
        for _, data in self.kg.graph.nodes(data=True):
            if 'reliability_rating' in data:
                rating_name = data['reliability_rating'].name
                summary[rating_name] += 1
                
        return summary
        
    def get_category_summary(self) -> Dict[str, int]:
        """
        Get a summary of fact categories in the knowledge graph.
        
        Returns:
            Dictionary mapping category names to counts
        """
        summary = {}
        
        for _, data in self.kg.graph.nodes(data=True):
            if 'category' in data:
                category = data['category']
                summary[category] = summary.get(category, 0) + 1
                
        return summary
        
    def find_contradictions(self) -> List[Tuple[str, str]]:
        """
        Find potentially contradicting facts based on similarity and reliability.
        
        This is a simple implementation that looks for facts with similar statements
        but different reliability ratings.
        
        Returns:
            List of tuples containing pairs of potentially contradicting fact IDs
        """
        contradictions = []
        facts = list(self.kg.graph.nodes(data=True))
        
        for i in range(len(facts)):
            node_i, data_i = facts[i]
            for j in range(i+1, len(facts)):
                node_j, data_j = facts[j]
                
                # Skip if facts don't have statements or reliability ratings
                if ('fact_statement' not in data_i or 
                    'fact_statement' not in data_j or
                    'reliability_rating' not in data_i or
                    'reliability_rating' not in data_j):
                    continue
                    
                # Check if statements are similar but reliability differs significantly
                statement_i = data_i['fact_statement'].lower()
                statement_j = data_j['fact_statement'].lower()
                
                # Simple similarity check - can be improved with NLP techniques
                if (self._simple_similarity(statement_i, statement_j) > 0.7 and
                    abs(data_i['reliability_rating'].value - data_j['reliability_rating'].value) >= 2):
                    contradictions.append((node_i, node_j))
                    
        return contradictions
        
    def get_fact_importance(self) -> Dict[str, float]:
        """
        Calculate importance score for each fact based on multiple factors.
        
        The importance score is based on:
        - Reliability rating
        - Usage count
        - Centrality in the graph
        - Number of related facts
        
        Returns:
            Dictionary mapping fact_id to importance score
        """
        importance_scores = {}
        
        # Calculate centrality
        if len(self.kg.graph) > 0:
            centrality = nx.betweenness_centrality(self.kg.graph)
        else:
            centrality = {}
            
        for node, data in self.kg.graph.nodes(data=True):
            # Base score from reliability and usage
            reliability_score = data.get('reliability_rating', ReliabilityRating.UNVERIFIED).value
            usage_score = data.get('usage_count', 0)
            
            # Graph structure factors
            centrality_score = centrality.get(node, 0) * 10  # Scale up centrality
            related_facts_score = len(data.get('related_facts', []))
            
            # Calculate total importance score
            importance = (
                reliability_score * 2 +  # Weight reliability more
                usage_score * 0.5 +
                centrality_score * 3 +   # Weight centrality more
                related_facts_score * 0.5
            )
            
            importance_scores[node] = importance
            
        return importance_scores
        
    def suggest_missing_links(self, threshold: float = 0.6) -> List[Tuple[str, str, float]]:
        """
        Suggest potentially missing links between facts based on common attributes.
        
        Args:
            threshold: Similarity threshold for suggesting links
            
        Returns:
            List of tuples containing (fact_id1, fact_id2, similarity_score)
        """
        suggestions = []
        facts = list(self.kg.graph.nodes(data=True))
        
        for i in range(len(facts)):
            node_i, data_i = facts[i]
            for j in range(i+1, len(facts)):
                node_j, data_j = facts[j]
                
                # Skip if already directly connected
                if self.kg.graph.has_edge(node_i, node_j) or self.kg.graph.has_edge(node_j, node_i):
                    continue
                    
                # Calculate similarity based on attributes
                similarity = self._calculate_fact_similarity(data_i, data_j)
                
                if similarity >= threshold:
                    suggestions.append((node_i, node_j, similarity))
                    
        # Sort by similarity score
        return sorted(suggestions, key=lambda x: x[2], reverse=True)
        
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two text strings.
        
        This is a basic implementation using word overlap.
        For production use, consider using more sophisticated NLP techniques.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
        
    def _calculate_fact_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two facts based on their attributes.
        
        Args:
            data1: Attributes of first fact
            data2: Attributes of second fact
            
        Returns:
            Similarity score between 0 and 1
        """
        similarity_scores = []
        
        # Compare fact statements
        if 'fact_statement' in data1 and 'fact_statement' in data2:
            statement_sim = self._simple_similarity(data1['fact_statement'], data2['fact_statement'])
            similarity_scores.append(statement_sim * 0.4)  # Weight statements more
            
        # Compare categories
        if 'category' in data1 and 'category' in data2:
            category_sim = 1.0 if data1['category'] == data2['category'] else 0.0
            similarity_scores.append(category_sim * 0.2)
            
        # Compare tags
        if 'tags' in data1 and 'tags' in data2:
            tags1 = set(data1['tags'].split(',')) if isinstance(data1['tags'], str) else set(data1['tags'])
            tags2 = set(data2['tags'].split(',')) if isinstance(data2['tags'], str) else set(data2['tags'])
            
            if tags1 and tags2:
                tags_sim = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
                similarity_scores.append(tags_sim * 0.2)
                
        # Compare sources
        if 'source_id' in data1 and 'source_id' in data2:
            source_sim = 1.0 if data1['source_id'] == data2['source_id'] else 0.0
            similarity_scores.append(source_sim * 0.1)
            
        # Compare authors
        if 'author_creator' in data1 and 'author_creator' in data2:
            author_sim = 1.0 if data1['author_creator'] == data2['author_creator'] else 0.0
            similarity_scores.append(author_sim * 0.1)
            
        # If no scores were calculated, return 0
        if not similarity_scores:
            return 0.0
            
        # Return weighted average of similarity scores
        return sum(similarity_scores)
