"""
Core functionality of the knowledge graph package.

This module provides the basic knowledge schema and features for creating and managing knowledge graphs.
It implements the KnowledgeGraph class which serves as the main interface for adding, retrieving,
and updating facts with reliability ratings.

Classes:
    ReliabilityRating: Enum for fact reliability levels
    KnowledgeGraph: Main class for knowledge graph operations

Example:
    from knowledge_graph_pkg import KnowledgeGraph
    from knowledge_graph_pkg.core import ReliabilityRating
    
    kg = KnowledgeGraph()
    kg.add_fact("fact1", "Earth is round", "Science", ["earth", "shape"],
               datetime.now(), datetime.now(), ReliabilityRating.VERIFIED,
               "source1", "Science Journal", "Researcher", datetime.now(),
               "https://example.com", [], "Notes", "public", 10)
"""
import networkx as nx
from datetime import datetime
from enum import Enum
from typing import Dict, List, Union, Optional, Any

class ReliabilityRating(Enum):
    """Enum representing the reliability rating of a fact.
    
    Attributes:
        UNVERIFIED (1): Fact has not been verified
        POSSIBLY_TRUE (2): Fact might be true but requires more verification
        LIKELY_TRUE (3): Fact is likely true based on reliable sources
        VERIFIED (4): Fact has been verified by multiple reliable sources
    """
    UNVERIFIED = 1
    POSSIBLY_TRUE = 2
    LIKELY_TRUE = 3
    VERIFIED = 4

class KnowledgeGraph:
    """A class for creating and managing knowledge graphs with reliability ratings.
    
    This class provides methods to add, retrieve, and update facts in a knowledge graph.
    Each fact can have metadata including reliability ratings, source information,
    and contextual notes.
    
    Attributes:
        graph: A NetworkX DiGraph object representing the knowledge graph
    """
    
    def __init__(self):
        """Initialize a new KnowledgeGraph with an empty NetworkX DiGraph."""
        self.graph = nx.DiGraph()
    
    def validate_fact_id(self, fact_id: str) -> None:
        """Validate that a fact ID is a non-empty string.
        
        Args:
            fact_id: The ID to validate
            
        Raises:
            ValueError: If fact_id is not a non-empty string
        """
        if not isinstance(fact_id, str) or not fact_id:
            raise ValueError("Fact ID must be a non-empty string.")
    
    def validate_reliability_rating(self, rating: ReliabilityRating) -> None:
        """Validate that a reliability rating is a valid ReliabilityRating enum.
        
        Args:
            rating: The reliability rating to validate
            
        Raises:
            ValueError: If rating is not a ReliabilityRating enum
        """
        if not isinstance(rating, ReliabilityRating):
            raise ValueError("Reliability rating must be an instance of ReliabilityRating Enum.")
    
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
        """Add a new fact to the knowledge graph.
        
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
        self.validate_fact_id(fact_id)
        self.validate_reliability_rating(reliability_rating)
        # Additional validations for other parameters can be added here
        
        try:
            # Conversion of list and datetime objects to strings for storage
            tags_str = ', '.join(tags) if tags else ''
            date_recorded_str = date_recorded.isoformat() if isinstance(date_recorded, datetime) else date_recorded
            last_updated_str = last_updated.isoformat() if isinstance(last_updated, datetime) else last_updated
            publication_date_str = publication_date.isoformat() if isinstance(publication_date, datetime) else publication_date
            
            # Calculate quality score based on reliability rating and usage count
            quality_score = reliability_rating.value * 10 + usage_count * 2
            
            # Adding fact to the graph
            self.graph.add_node(fact_id, 
                               fact_statement=fact_statement, 
                               category=category,
                               tags=tags_str, 
                               date_recorded=date_recorded_str, 
                               last_updated=last_updated_str,
                               reliability_rating=reliability_rating, 
                               source_id=source_id, 
                               source_title=source_title,
                               author_creator=author_creator, 
                               publication_date=publication_date_str,
                               url_reference=url_reference, 
                               related_facts=related_facts, 
                               contextual_notes=contextual_notes,
                               access_level=access_level, 
                               usage_count=usage_count,
                               quality_score=quality_score)
        except Exception as e:
            raise Exception(f"Error adding fact: {e}")
    
    def get_fact(self, fact_id: str) -> Dict[str, Any]:
        """Retrieve a fact from the knowledge graph.
        
        Args:
            fact_id: The ID of the fact to retrieve
            
        Returns:
            A dictionary containing the fact's attributes
            
        Raises:
            ValueError: If fact_id is invalid or not found
        """
        self.validate_fact_id(fact_id)
        if fact_id not in self.graph:
            raise ValueError(f"Fact ID '{fact_id}' not found in the graph.")
        return self.graph.nodes[fact_id]
    
    def update_fact(self, fact_id: str, **kwargs) -> None:
        """Update attributes of an existing fact.
        
        Args:
            fact_id: The ID of the fact to update
            **kwargs: Keyword arguments representing attributes to update
            
        Raises:
            ValueError: If fact_id is invalid or not found, or if an invalid attribute is specified
            Exception: If there's an error updating the fact
        """
        self.validate_fact_id(fact_id)
        if fact_id not in self.graph:
            raise ValueError(f"Fact ID '{fact_id}' not found in the graph.")
        
        try:
            # If reliability_rating or usage_count is updated, recalculate quality_score
            if 'reliability_rating' in kwargs or 'usage_count' in kwargs:
                reliability_rating = kwargs.get('reliability_rating', self.graph.nodes[fact_id]['reliability_rating'])
                usage_count = kwargs.get('usage_count', self.graph.nodes[fact_id]['usage_count'])
                kwargs['quality_score'] = reliability_rating.value * 10 + usage_count * 2
            
            for key, value in kwargs.items():
                if key in self.graph.nodes[fact_id]:
                    self.graph.nodes[fact_id][key] = value
                else:
                    raise ValueError(f"Invalid attribute '{key}' for fact update.")
        except Exception as e:
            raise Exception(f"Error updating fact: {e}")
