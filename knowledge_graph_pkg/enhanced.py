"""
Enhanced knowledge graph operations with caching and performance optimizations.

This module provides an enhanced version of the KnowledgeGraph class with
additional features for performance, caching, and advanced operations.
"""

import networkx as nx
import json
import os
import time
from typing import Dict, List, Any, Union, Optional, Tuple, Set, Callable
from datetime import datetime
from functools import lru_cache
from .core import KnowledgeGraph, ReliabilityRating


class EnhancedKnowledgeGraph(KnowledgeGraph):
    """
    Enhanced version of KnowledgeGraph with performance optimizations and advanced features.
    
    This class extends the base KnowledgeGraph with caching, batch operations,
    change tracking, and performance optimizations.
    
    Attributes:
        graph: A NetworkX DiGraph object representing the knowledge graph
        cache_enabled: Whether caching is enabled
        _changes: List of changes made to the graph for tracking
        _last_save: Timestamp of last save operation
    """
    
    def __init__(self, cache_enabled: bool = True, auto_save_interval: int = 300):
        """
        Initialize an EnhancedKnowledgeGraph with optional caching.
        
        Args:
            cache_enabled: Whether to enable caching for improved performance
            auto_save_interval: Interval in seconds for auto-saving (0 to disable)
        """
        super().__init__()
        self.cache_enabled = cache_enabled
        self.auto_save_interval = auto_save_interval
        self._changes = []
        self._last_save = time.time()
        
    def add_fact(self, 
                fact_id: str, 
                fact_statement: str, 
                category: str, 
                tags: List[str], 
                date_recorded: Union[datetime, str], 
                last_updated: Union[datetime, str],
                reliability_rating: ReliabilityRating, 
                source_id: str, 
                source_title: str, 
                author_creator: str,
                publication_date: Union[datetime, str], 
                url_reference: str, 
                related_facts: List[str], 
                contextual_notes: str,
                access_level: str, 
                usage_count: int) -> None:
        """
        Add a new fact to the knowledge graph with change tracking.
        
        Args:
            fact_id: Unique identifier for the fact
            fact_statement: The actual fact statement
            category: Category of the fact (e.g., Science, History)
            tags: List of tags associated with the fact
            date_recorded: Date when the fact was recorded
            last_updated: Date when the fact was last updated
            reliability_rating: ReliabilityRating enum value
            source_id: Identifier for the source
            source_title: Title of the source
            author_creator: Author or creator of the fact
            publication_date: Date of publication
            url_reference: URL reference for the fact
            related_facts: List of related fact IDs
            contextual_notes: Additional notes about the fact
            access_level: Access level for the fact (e.g., public, private)
            usage_count: Number of times the fact has been used
            
        Raises:
            ValueError: If validation fails for any parameter
            Exception: If there's an error adding the fact to the graph
        """
        # Call the parent method to add the fact
        super().add_fact(
            fact_id, fact_statement, category, tags, date_recorded, last_updated,
            reliability_rating, source_id, source_title, author_creator,
            publication_date, url_reference, related_facts, contextual_notes,
            access_level, usage_count
        )
        
        # Track the change
        self._track_change("add", fact_id)
        
        # Check if auto-save is needed
        self._check_auto_save()
        
        # Clear relevant caches
        if self.cache_enabled:
            self.get_facts_by_category.cache_clear()
            self.get_facts_by_reliability.cache_clear()
            
    def update_fact(self, fact_id: str, **kwargs) -> None:
        """
        Update attributes of an existing fact with change tracking.
        
        Args:
            fact_id: The ID of the fact to update
            **kwargs: Keyword arguments representing attributes to update
            
        Raises:
            ValueError: If fact_id is invalid or not found, or if an invalid attribute is specified
            Exception: If there's an error updating the fact
        """
        # Call the parent method to update the fact
        super().update_fact(fact_id, **kwargs)
        
        # Track the change
        self._track_change("update", fact_id, kwargs)
        
        # Check if auto-save is needed
        self._check_auto_save()
        
        # Clear relevant caches
        if self.cache_enabled:
            self.get_facts_by_category.cache_clear()
            self.get_facts_by_reliability.cache_clear()
            self.get_fact.cache_clear()
            
    @lru_cache(maxsize=128)
    def get_fact(self, fact_id: str) -> Dict[str, Any]:
        """
        Retrieve a fact from the knowledge graph with caching.
        
        Args:
            fact_id: The ID of the fact to retrieve
            
        Returns:
            A dictionary containing the fact's attributes
            
        Raises:
            ValueError: If fact_id is invalid or not found
        """
        return super().get_fact(fact_id)
        
    def delete_fact(self, fact_id: str) -> None:
        """
        Delete a fact from the knowledge graph.
        
        Args:
            fact_id: The ID of the fact to delete
            
        Raises:
            ValueError: If fact_id is invalid or not found
        """
        self.validate_fact_id(fact_id)
        if fact_id not in self.graph:
            raise ValueError(f"Fact ID '{fact_id}' not found in the graph.")
            
        # Remove the fact
        self.graph.remove_node(fact_id)
        
        # Track the change
        self._track_change("delete", fact_id)
        
        # Check if auto-save is needed
        self._check_auto_save()
        
        # Clear relevant caches
        if self.cache_enabled:
            self.get_facts_by_category.cache_clear()
            self.get_facts_by_reliability.cache_clear()
            self.get_fact.cache_clear()
            
    def batch_add_facts(self, facts: List[Dict[str, Any]]) -> int:
        """
        Add multiple facts to the knowledge graph in a single batch operation.
        
        Args:
            facts: List of dictionaries containing fact attributes
            
        Returns:
            Number of facts successfully added
            
        Raises:
            Exception: If there's an error adding facts
        """
        added_count = 0
        
        for fact_data in facts:
            try:
                fact_id = fact_data.pop('fact_id')
                self.add_fact(fact_id=fact_id, **fact_data)
                added_count += 1
            except Exception as e:
                print(f"Error adding fact: {e}")
                
        return added_count
        
    @lru_cache(maxsize=32)
    def get_facts_by_category(self, category: str) -> List[str]:
        """
        Get all facts in a specific category with caching.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of fact IDs in the specified category
        """
        facts = []
        
        for node, data in self.graph.nodes(data=True):
            if data.get('category') == category:
                facts.append(node)
                
        return facts
        
    @lru_cache(maxsize=32)
    def get_facts_by_reliability(self, min_reliability: ReliabilityRating) -> List[str]:
        """
        Get all facts with at least the specified reliability rating with caching.
        
        Args:
            min_reliability: Minimum reliability rating to include
            
        Returns:
            List of fact IDs meeting the reliability criteria
        """
        facts = []
        
        for node, data in self.graph.nodes(data=True):
            if data.get('reliability_rating', ReliabilityRating.UNVERIFIED).value >= min_reliability.value:
                facts.append(node)
                
        return facts
        
    def search_facts(self, query: str, fields: List[str] = None) -> List[Tuple[str, float]]:
        """
        Search for facts matching a query string.
        
        Args:
            query: Search query string
            fields: List of fields to search in (defaults to ['fact_statement', 'contextual_notes'])
            
        Returns:
            List of tuples containing (fact_id, relevance_score)
        """
        if fields is None:
            fields = ['fact_statement', 'contextual_notes']
            
        results = []
        query_terms = query.lower().split()
        
        for node, data in self.graph.nodes(data=True):
            score = 0
            
            for field in fields:
                if field in data:
                    field_value = str(data[field]).lower()
                    
                    # Calculate simple relevance score based on term matches
                    for term in query_terms:
                        if term in field_value:
                            score += 1
                            
                    # Bonus for exact phrase match
                    if query.lower() in field_value:
                        score += len(query_terms)
                        
            if score > 0:
                results.append((node, score))
                
        # Sort by relevance score
        return sorted(results, key=lambda x: x[1], reverse=True)
        
    def save_to_file(self, filename: str) -> None:
        """
        Save the knowledge graph to a file with change tracking.
        
        Args:
            filename: Path to save the file
            
        Raises:
            Exception: If there's an error saving the file
        """
        data = {}
        
        for node, node_data in self.graph.nodes(data=True):
            # Convert data for serialization
            serialized_data = dict(node_data)
            
            # Convert enum to string
            if 'reliability_rating' in serialized_data:
                serialized_data['reliability_rating'] = serialized_data['reliability_rating'].name
                
            # Convert datetime objects to ISO format strings
            for key, value in serialized_data.items():
                if isinstance(value, datetime):
                    serialized_data[key] = value.isoformat()
                    
            data[node] = serialized_data
            
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        # Update last save time
        self._last_save = time.time()
        
        # Clear change log
        self._changes = []
        
    def load_from_file(self, filename: str) -> int:
        """
        Load the knowledge graph from a file.
        
        Args:
            filename: Path to the file to load
            
        Returns:
            Number of facts loaded
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is invalid JSON
        """
        # Clear existing graph
        self.graph = nx.DiGraph()
        
        # Clear caches
        if self.cache_enabled:
            self.get_fact.cache_clear()
            self.get_facts_by_category.cache_clear()
            self.get_facts_by_reliability.cache_clear()
            
        # Load from file
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        loaded_count = 0
        
        for fact_id, fact_data in data.items():
            try:
                # Convert string reliability rating to enum
                if 'reliability_rating' in fact_data and isinstance(fact_data['reliability_rating'], str):
                    fact_data['reliability_rating'] = getattr(ReliabilityRating, fact_data['reliability_rating'])
                    
                # Add the fact without tracking changes
                super().add_fact(fact_id=fact_id, **fact_data)
                loaded_count += 1
                
            except Exception as e:
                print(f"Error loading fact {fact_id}: {e}")
                
        # Update last save time
        self._last_save = time.time()
        
        # Clear change log
        self._changes = []
        
        return loaded_count
        
    def get_changes(self) -> List[Dict[str, Any]]:
        """
        Get the list of changes made to the knowledge graph.
        
        Returns:
            List of change dictionaries
        """
        return self._changes
        
    def clear_caches(self) -> None:
        """
        Clear all method caches.
        """
        if self.cache_enabled:
            self.get_fact.cache_clear()
            self.get_facts_by_category.cache_clear()
            self.get_facts_by_reliability.cache_clear()
            
    def _track_change(self, operation: str, fact_id: str, details: Dict[str, Any] = None) -> None:
        """
        Track a change to the knowledge graph.
        
        Args:
            operation: Type of operation (add, update, delete)
            fact_id: ID of the affected fact
            details: Additional details about the change
        """
        change = {
            'operation': operation,
            'fact_id': fact_id,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            change['details'] = details
            
        self._changes.append(change)
        
    def _check_auto_save(self) -> None:
        """
        Check if auto-save should be triggered and save if needed.
        """
        if self.auto_save_interval > 0 and (time.time() - self._last_save) > self.auto_save_interval:
            # Auto-save to a timestamped file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"autosave_knowledge_graph_{timestamp}.json"
            
            try:
                self.save_to_file(filename)
                print(f"Auto-saved knowledge graph to {filename}")
            except Exception as e:
                print(f"Error during auto-save: {e}")
