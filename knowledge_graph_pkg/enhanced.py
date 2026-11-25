"""
Enhanced knowledge graph operations with caching and performance optimizations.

This module provides an enhanced version of the KnowledgeGraph class with
additional features for performance, caching, and advanced operations.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from .core import KnowledgeGraph, ReliabilityRating


def _parse_reliability(val: Any) -> ReliabilityRating:
    if isinstance(val, ReliabilityRating):
        return val
    if isinstance(val, str):
        member = ReliabilityRating.__members__.get(val.strip().upper())
        if member is not None:
            return member
    raise ValueError(f"Invalid reliability rating: {val!r}")


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
    
    def __init__(self, cache_enabled: bool = True, auto_save_interval: int = 300,
                 auto_save_dir: Optional[str] = None):
        """
        Initialize an EnhancedKnowledgeGraph with optional caching.
        
        Args:
            cache_enabled: Whether to enable caching for improved performance
            auto_save_interval: Interval in seconds for auto-saving (0 to disable)
            auto_save_dir: Optional directory for auto-save output files
        """
        super().__init__()
        self.cache_enabled = cache_enabled
        self.auto_save_interval = auto_save_interval
        self.auto_save_dir = auto_save_dir
        self._changes = []
        self._last_save = time.time()
        self._fact_cache: Dict[str, Dict[str, Any]] = {}
        self._category_cache: Dict[str, List[str]] = {}
        self._reliability_cache: Dict[ReliabilityRating, List[str]] = {}
        
    def add_fact(self, 
                fact_id: str, 
                fact_statement: str, 
                category: str, 
                tags: List[str], 
                date_recorded,
                last_updated,
                reliability_rating: ReliabilityRating, 
                source_id: str, 
                source_title: str, 
                author_creator: str,
                publication_date,
                url_reference: str, 
                related_facts: List[str], 
                contextual_notes: str,
                access_level: str, 
                usage_count: int) -> None:
        """
        Add a new fact to the knowledge graph with change tracking.
        """
        super().add_fact(
            fact_id, fact_statement, category, tags, date_recorded, last_updated,
            reliability_rating, source_id, source_title, author_creator,
            publication_date, url_reference, related_facts, contextual_notes,
            access_level, usage_count
        )

        self.clear_caches()
        self._track_change("add", fact_id)
        self._check_auto_save()
            
    def update_fact(self, fact_id: str, **kwargs) -> None:
        """
        Update attributes of an existing fact with change tracking.
        """
        super().update_fact(fact_id, **kwargs)

        self.clear_caches()
        self._track_change("update", fact_id, dict(kwargs))
        self._check_auto_save()
            
    def get_fact(self, fact_id: str) -> Dict[str, Any]:
        """
        Retrieve a fact from the knowledge graph with per-instance caching.
        """
        if self.cache_enabled and fact_id in self._fact_cache:
            return self._fact_cache[fact_id]
        fact = super().get_fact(fact_id)
        if self.cache_enabled:
            self._fact_cache[fact_id] = fact
        return fact
        
    def delete_fact(self, fact_id: str) -> None:
        """
        Delete a fact from the knowledge graph.
        """
        self.validate_fact_id(fact_id)
        if fact_id not in self.graph:
            raise ValueError(f"Fact ID '{fact_id}' not found in the graph.")
            
        self.graph.remove_node(fact_id)
        self.clear_caches()
        self._track_change("delete", fact_id)
        self._check_auto_save()
            
    def batch_add_facts(self, facts: List[Dict[str, Any]]) -> int:
        """
        Add multiple facts to the knowledge graph in a single batch operation.
        """
        added_count = 0
        
        for raw_fact in facts:
            try:
                fact_data = dict(raw_fact)
                fact_id = fact_data.pop('fact_id')
                fact_data.pop('quality_score', None)
                if isinstance(fact_data.get('tags'), str):
                    fact_data['tags'] = [
                        t.strip() for t in fact_data['tags'].split(',') if t.strip()
                    ]
                if isinstance(fact_data.get('reliability_rating'), str):
                    fact_data['reliability_rating'] = _parse_reliability(
                        fact_data['reliability_rating']
                    )
                self.add_fact(fact_id=fact_id, **fact_data)
                added_count += 1
            except Exception as e:
                print(f"Error adding fact: {e}")
                
        return added_count
        
    def get_facts_by_category(self, category: str) -> List[str]:
        """
        Get all facts in a specific category with caching.
        """
        if self.cache_enabled and category in self._category_cache:
            return list(self._category_cache[category])
        facts = [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get('category') == category
        ]
        if self.cache_enabled:
            self._category_cache[category] = list(facts)
        return facts
        
    def get_facts_by_reliability(self, min_reliability: ReliabilityRating) -> List[str]:
        """
        Get all facts with at least the specified reliability rating with caching.
        """
        if self.cache_enabled and min_reliability in self._reliability_cache:
            return list(self._reliability_cache[min_reliability])
        facts = []
        for node, data in self.graph.nodes(data=True):
            raw_rel = data.get('reliability_rating', ReliabilityRating.UNVERIFIED)
            rel_val = raw_rel.value if isinstance(raw_rel, ReliabilityRating) else 1
            if rel_val >= min_reliability.value:
                facts.append(node)
        if self.cache_enabled:
            self._reliability_cache[min_reliability] = list(facts)
        return facts
        
    def search_facts(self, query: str, fields: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Search for facts matching a query string.
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
                    
                    for term in query_terms:
                        if term in field_value:
                            score += 1
                            
                    if query.lower() in field_value:
                        score += len(query_terms)
                        
            if score > 0:
                results.append((node, score))
                
        return sorted(results, key=lambda x: x[1], reverse=True)
        
    def save_to_file(self, filename: str) -> None:
        """
        Save the knowledge graph to a file atomically with change tracking.
        """
        data = {}
        
        for node, node_data in self.graph.nodes(data=True):
            serialized_data = dict(node_data)
            
            if 'reliability_rating' in serialized_data:
                rel = serialized_data['reliability_rating']
                serialized_data['reliability_rating'] = (
                    rel.name if isinstance(rel, ReliabilityRating) else str(rel)
                )
                
            for key, value in serialized_data.items():
                if isinstance(value, datetime):
                    serialized_data[key] = value.isoformat()
                    
            data[node] = serialized_data

        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{filename}.tmp.{os.getpid()}"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filename)
            
        self._last_save = time.time()
        self._changes = []
        
    def load_from_file(self, filename: str) -> int:
        """
        Load the knowledge graph from a file.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.graph = nx.DiGraph()
        self.clear_caches()
            
        loaded_count = 0
        
        for fact_id, raw_fact in data.items():
            try:
                fact_data = dict(raw_fact)
                fact_data.pop('quality_score', None)
                if isinstance(fact_data.get('tags'), str):
                    fact_data['tags'] = [
                        t.strip() for t in fact_data['tags'].split(',') if t.strip()
                    ]
                if isinstance(fact_data.get('reliability_rating'), str):
                    fact_data['reliability_rating'] = _parse_reliability(
                        fact_data['reliability_rating']
                    )
                    
                super().add_fact(fact_id=fact_id, **fact_data)
                loaded_count += 1
                
            except Exception as e:
                print(f"Error loading fact {fact_id}: {e}")
                
        self._last_save = time.time()
        self._changes = []
        
        return loaded_count
        
    def get_changes(self) -> List[Dict[str, Any]]:
        """
        Get the list of changes made to the knowledge graph.
        """
        return self._changes
        
    def clear_caches(self) -> None:
        """
        Clear all per-instance method caches.
        """
        self._fact_cache.clear()
        self._category_cache.clear()
        self._reliability_cache.clear()
            
    def _track_change(self, operation: str, fact_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Track a change to the knowledge graph.
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"autosave_knowledge_graph_{timestamp}.json"
            filename = os.path.join(self.auto_save_dir, name) if self.auto_save_dir else name
            
            try:
                self.save_to_file(filename)
                print(f"Auto-saved knowledge graph to {filename}")
            except Exception as e:
                print(f"Error during auto-save: {e}")
