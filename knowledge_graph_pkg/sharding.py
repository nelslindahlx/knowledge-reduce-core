"""
Sharding and distributed knowledge graph capabilities.

This module provides functionality for sharding large knowledge graphs
across multiple files or databases, enabling efficient handling of
very large knowledge bases.
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Union, Optional, Tuple, Set, Callable
from datetime import datetime
from .core import KnowledgeGraph, ReliabilityRating

class ShardedKnowledgeGraph:
    """
    Class for managing sharded knowledge graphs.
    
    This class provides methods for distributing a knowledge graph across
    multiple shards for improved performance with large datasets.
    
    Attributes:
        base_dir: Directory to store shards
        shard_size: Maximum number of facts per shard
        shards: Dictionary mapping shard IDs to KnowledgeGraph instances
        fact_to_shard: Dictionary mapping fact IDs to shard IDs
    """
    
    def __init__(self, base_dir: str, shard_size: int = 1000):
        """
        Initialize a ShardedKnowledgeGraph.
        
        Args:
            base_dir: Directory to store shards
            shard_size: Maximum number of facts per shard
        """
        self.base_dir = base_dir
        self.shard_size = shard_size
        self.shards = {}
        self.fact_to_shard = {}
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Load existing shards if any
        self._load_existing_shards()
        
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
                usage_count: int) -> str:
        """
        Add a new fact to the sharded knowledge graph.
        
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
            
        Returns:
            Shard ID where the fact was added
            
        Raises:
            ValueError: If validation fails for any parameter
            Exception: If there's an error adding the fact
        """
        # Determine which shard to use
        shard_id = self._get_shard_for_fact(fact_id)
        
        # Get or create the shard
        if shard_id not in self.shards:
            self.shards[shard_id] = KnowledgeGraph()
        
        # Add the fact to the shard
        self.shards[shard_id].add_fact(
            fact_id, fact_statement, category, tags, date_recorded, last_updated,
            reliability_rating, source_id, source_title, author_creator,
            publication_date, url_reference, related_facts, contextual_notes,
            access_level, usage_count
        )
        
        # Map the fact to its shard
        self.fact_to_shard[fact_id] = shard_id
        
        # Save the shard
        self._save_shard(shard_id)
        
        # Save the fact-to-shard mapping
        self._save_fact_mapping()
        
        return shard_id
        
    def get_fact(self, fact_id: str) -> Dict[str, Any]:
        """
        Retrieve a fact from the sharded knowledge graph.
        
        Args:
            fact_id: The ID of the fact to retrieve
            
        Returns:
            A dictionary containing the fact's attributes
            
        Raises:
            ValueError: If fact_id is invalid or not found
        """
        # Find which shard contains the fact
        if fact_id not in self.fact_to_shard:
            raise ValueError(f"Fact ID '{fact_id}' not found in any shard.")
            
        shard_id = self.fact_to_shard[fact_id]
        
        # Load the shard if not already loaded
        if shard_id not in self.shards:
            self._load_shard(shard_id)
            
        # Get the fact from the shard
        return self.shards[shard_id].get_fact(fact_id)
        
    def update_fact(self, fact_id: str, **kwargs) -> None:
        """
        Update attributes of an existing fact in the sharded knowledge graph.
        
        Args:
            fact_id: The ID of the fact to update
            **kwargs: Keyword arguments representing attributes to update
            
        Raises:
            ValueError: If fact_id is invalid or not found, or if an invalid attribute is specified
            Exception: If there's an error updating the fact
        """
        # Find which shard contains the fact
        if fact_id not in self.fact_to_shard:
            raise ValueError(f"Fact ID '{fact_id}' not found in any shard.")
            
        shard_id = self.fact_to_shard[fact_id]
        
        # Load the shard if not already loaded
        if shard_id not in self.shards:
            self._load_shard(shard_id)
            
        # Update the fact in the shard
        self.shards[shard_id].update_fact(fact_id, **kwargs)
        
        # Save the shard
        self._save_shard(shard_id)
        
    def delete_fact(self, fact_id: str) -> None:
        """
        Delete a fact from the sharded knowledge graph.
        
        Args:
            fact_id: The ID of the fact to delete
            
        Raises:
            ValueError: If fact_id is invalid or not found
        """
        # Find which shard contains the fact
        if fact_id not in self.fact_to_shard:
            raise ValueError(f"Fact ID '{fact_id}' not found in any shard.")
            
        shard_id = self.fact_to_shard[fact_id]
        
        # Load the shard if not already loaded
        if shard_id not in self.shards:
            self._load_shard(shard_id)
            
        # Get the shard's graph
        graph = self.shards[shard_id].graph
        
        # Validate the fact exists
        if fact_id not in graph:
            raise ValueError(f"Fact ID '{fact_id}' not found in shard {shard_id}.")
            
        # Remove the fact from the shard
        graph.remove_node(fact_id)
        
        # Remove the fact from the mapping
        del self.fact_to_shard[fact_id]
        
        # Save the shard
        self._save_shard(shard_id)
        
        # Save the fact-to-shard mapping
        self._save_fact_mapping()
        
    def search_facts(self, query: Dict[str, Any]) -> List[str]:
        """
        Search for facts matching query criteria across all shards.
        
        Args:
            query: Dictionary of attribute-value pairs to match
            
        Returns:
            List of matching fact IDs
        """
        matching_facts = []
        
        # Search each shard
        for shard_id in self._get_all_shard_ids():
            # Load the shard if not already loaded
            if shard_id not in self.shards:
                self._load_shard(shard_id)
                
            # Search the shard
            for node, data in self.shards[shard_id].graph.nodes(data=True):
                matches = True
                
                for key, value in query.items():
                    if key not in data or data[key] != value:
                        matches = False
                        break
                        
                if matches:
                    matching_facts.append(node)
                    
        return matching_facts
        
    def get_facts_by_category(self, category: str) -> List[str]:
        """
        Get all facts in a specific category across all shards.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of fact IDs in the specified category
        """
        return self.search_facts({'category': category})
        
    def get_facts_by_reliability(self, min_reliability: ReliabilityRating) -> List[str]:
        """
        Get all facts with at least the specified reliability rating across all shards.
        
        Args:
            min_reliability: Minimum reliability rating to include
            
        Returns:
            List of fact IDs meeting the reliability criteria
        """
        matching_facts = []
        
        # Search each shard
        for shard_id in self._get_all_shard_ids():
            # Load the shard if not already loaded
            if shard_id not in self.shards:
                self._load_shard(shard_id)
                
            # Search the shard
            for node, data in self.shards[shard_id].graph.nodes(data=True):
                if data.get('reliability_rating', ReliabilityRating.UNVERIFIED).value >= min_reliability.value:
                    matching_facts.append(node)
                    
        return matching_facts
        
    def get_fact_count(self) -> int:
        """
        Get the total number of facts across all shards.
        
        Returns:
            Total number of facts
        """
        return len(self.fact_to_shard)
        
    def get_shard_count(self) -> int:
        """
        Get the number of shards.
        
        Returns:
            Number of shards
        """
        return len(self._get_all_shard_ids())
        
    def optimize_shards(self) -> Dict[str, int]:
        """
        Optimize shards by redistributing facts to maintain balanced shard sizes.
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = {
            'shards_before': self.get_shard_count(),
            'facts_moved': 0,
            'shards_after': 0
        }
        
        # Get all facts
        all_facts = {}
        for fact_id, shard_id in self.fact_to_shard.items():
            # Load the shard if not already loaded
            if shard_id not in self.shards:
                self._load_shard(shard_id)
                
            # Get the fact data
            all_facts[fact_id] = self.shards[shard_id].get_fact(fact_id)
            
        # Clear existing shards
        self.shards = {}
        self.fact_to_shard = {}
        
        # Redistribute facts
        for fact_id, fact_data in all_facts.items():
            # Add the fact to a new shard
            new_shard_id = self._get_shard_for_fact(fact_id)
            
            # Create new shard if needed
            if new_shard_id not in self.shards:
                self.shards[new_shard_id] = KnowledgeGraph()
                
            # Add fact to new shard
            self.shards[new_shard_id].graph.add_node(fact_id, **fact_data)
            
            # Map fact to new shard
            self.fact_to_shard[fact_id] = new_shard_id
            
            stats['facts_moved'] += 1
            
        # Save all shards
        for shard_id in self.shards:
            self._save_shard(shard_id)
            
        # Save fact mapping
        self._save_fact_mapping()
        
        stats['shards_after'] = self.get_shard_count()
        
        return stats
        
    def _get_shard_for_fact(self, fact_id: str) -> str:
        """
        Determine which shard a fact should be stored in.
        
        Args:
            fact_id: ID of the fact
            
        Returns:
            Shard ID
        """
        # If fact already has a shard, use that
        if fact_id in self.fact_to_shard:
            return self.fact_to_shard[fact_id]
            
        # Hash the fact ID to determine the shard
        hash_obj = hashlib.md5(fact_id.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Get existing shards
        existing_shards = self._get_all_shard_ids()
        
        # If no shards exist or all are full, create a new one
        if not existing_shards or all(self._is_shard_full(shard_id) for shard_id in existing_shards):
            if not existing_shards:
                new_shard_id = "shard_0001"
            else:
                # Get highest shard number and increment
                highest = max(int(shard_id.split('_')[1]) for shard_id in existing_shards)
                new_shard_id = f"shard_{highest+1:04d}"
                
            return new_shard_id
            
        # Find shards that aren't full
        available_shards = [shard_id for shard_id in existing_shards if not self._is_shard_full(shard_id)]
        
        # Use hash to select from available shards
        selected_index = hash_int % len(available_shards)
        return available_shards[selected_index]
        
    def _is_shard_full(self, shard_id: str) -> bool:
        """
        Check if a shard is full.
        
        Args:
            shard_id: ID of the shard to check
            
        Returns:
            True if shard is full, False otherwise
        """
        # Load the shard if not already loaded
        if shard_id not in self.shards:
            self._load_shard(shard_id)
            
        # Check if shard has reached maximum size
        return len(self.shards[shard_id].graph) >= self.shard_size
        
    def _load_shard(self, shard_id: str) -> None:
        """
        Load a shard from disk.
        
        Args:
            shard_id: ID of the shard to load
        """
        shard_path = os.path.join(self.base_dir, f"{shard_id}.json")
        
        if not os.path.exists(shard_path):
            # Create a new empty shard
            self.shards[shard_id] = KnowledgeGraph()
            return
            
        # Load the shard from file
        with open(shard_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create a new knowledge graph for the shard
        kg = KnowledgeGraph()
        
        # Add facts to the knowledge graph
        for fact_id, fact_data in data.items():
            # Convert string reliability rating to enum
            if 'reliability_rating' in fact_data and isinstance(fact_data['reliability_rating'], str):
                fact_data['reliability_rating'] = getattr(ReliabilityRating, fact_data['reliability_rating'])
                
            # Add the fact directly to the graph
            kg.graph.add_node(fact_id, **fact_data)
            
        # Store the shard
        self.shards[shard_id] = kg
        
    def _save_shard(self, shard_id: str) -> None:
        """
        Save a shard to disk.
        
        Args:
            shard_id: ID of the shard to save
        """
        shard_path = os.path.join(self.base_dir, f"{shard_id}.json")
        
        # Get the shard
        kg = self.shards[shard_id]
        
        # Prepare data for serialization
        data = {}
        for node, node_data in kg.graph.nodes(data=True):
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
        with open(shard_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def _save_fact_mapping(self) -> None:
        """
        Save the fact-to-shard mapping to disk.
        """
        mapping_path = os.path.join(self.base_dir, "fact_mapping.json")
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.fact_to_shard, f, indent=2)
            
    def _load_existing_shards(self) -> None:
        """
        Load existing shards and fact mapping from disk.
        """
        # Load fact mapping if it exists
        mapping_path = os.path.join(self.base_dir, "fact_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.fact_to_shard = json.load(f)
                
    def _get_all_shard_ids(self) -> List[str]:
        """
        Get all shard IDs from disk.
        
        Returns:
            List of shard IDs
        """
        shard_ids = []
        
        # Look for shard files in the base directory
        for filename in os.listdir(self.base_dir):
            if filename.startswith("shard_") and filename.endswith(".json"):
                shard_id = filename[:-5]  # Remove .json extension
                shard_ids.append(shard_id)
                
        return shard_ids
