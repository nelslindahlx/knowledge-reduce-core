"""
Utility functions for knowledge graph operations.

This module provides helper functions for common knowledge graph operations
such as importing data from various sources, filtering, and data transformation.
"""

import json
import csv
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
from .core import KnowledgeGraph, ReliabilityRating


def import_from_json(kg: KnowledgeGraph, json_file: str) -> int:
    """
    Import facts from a JSON file into a knowledge graph.
    
    Args:
        kg: KnowledgeGraph instance to import into
        json_file: Path to JSON file containing facts
        
    Returns:
        Number of facts successfully imported
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is invalid
    """
    imported_count = 0
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for fact_id, fact_data in data.items():
        try:
            # Convert string reliability rating to enum
            if 'reliability_rating' in fact_data and isinstance(fact_data['reliability_rating'], str):
                fact_data['reliability_rating'] = getattr(ReliabilityRating, fact_data['reliability_rating'])
                
            # Handle date fields
            for date_field in ['date_recorded', 'last_updated', 'publication_date']:
                if date_field in fact_data and isinstance(fact_data[date_field], str):
                    try:
                        fact_data[date_field] = datetime.fromisoformat(fact_data[date_field])
                    except ValueError:
                        # Keep as string if parsing fails
                        pass
            
            # Convert tags from string to list if needed
            if 'tags' in fact_data and isinstance(fact_data['tags'], str):
                fact_data['tags'] = [tag.strip() for tag in fact_data['tags'].split(',')]
                
            # Add the fact to the knowledge graph
            kg.add_fact(fact_id=fact_id, **fact_data)
            imported_count += 1
            
        except Exception as e:
            print(f"Error importing fact {fact_id}: {e}")
            
    return imported_count


def export_to_json(kg: KnowledgeGraph, json_file: str) -> int:
    """
    Export facts from a knowledge graph to a JSON file.
    
    Args:
        kg: KnowledgeGraph instance to export from
        json_file: Path to output JSON file
        
    Returns:
        Number of facts successfully exported
    """
    export_data = {}
    
    for node, data in kg.graph.nodes(data=True):
        node_data = dict(data)
        
        # Convert enum to string
        if 'reliability_rating' in node_data:
            node_data['reliability_rating'] = node_data['reliability_rating'].name
            
        # Convert datetime objects to ISO format strings
        for key, value in node_data.items():
            if isinstance(value, datetime):
                node_data[key] = value.isoformat()
                
        export_data[node] = node_data
        
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
        
    return len(export_data)


def import_from_csv(kg: KnowledgeGraph, csv_file: str, 
                   id_column: str = 'fact_id',
                   reliability_column: Optional[str] = 'reliability_rating') -> int:
    """
    Import facts from a CSV file into a knowledge graph.
    
    Args:
        kg: KnowledgeGraph instance to import into
        csv_file: Path to CSV file containing facts
        id_column: Name of the column containing fact IDs
        reliability_column: Name of the column containing reliability ratings
        
    Returns:
        Number of facts successfully imported
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    imported_count = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                fact_id = row.pop(id_column)
                
                # Convert reliability rating string to enum if present
                if reliability_column and reliability_column in row:
                    try:
                        row[reliability_column] = getattr(ReliabilityRating, row[reliability_column])
                    except (AttributeError, TypeError):
                        # Default to UNVERIFIED if conversion fails
                        row[reliability_column] = ReliabilityRating.UNVERIFIED
                
                # Add the fact to the knowledge graph
                kg.add_fact(fact_id=fact_id, **row)
                imported_count += 1
                
            except Exception as e:
                print(f"Error importing row: {e}")
                
    return imported_count


def extract_facts_from_url(url: str, 
                          fact_extractor: callable,
                          reliability: ReliabilityRating = ReliabilityRating.UNVERIFIED) -> List[Dict[str, Any]]:
    """
    Extract facts from a web URL using a custom extractor function.
    
    Args:
        url: URL to extract facts from
        fact_extractor: Function that takes BeautifulSoup object and returns list of fact dictionaries
        reliability: Default reliability rating for extracted facts
        
    Returns:
        List of extracted facts as dictionaries
        
    Raises:
        requests.RequestException: If there's an error fetching the URL
    """
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    facts = fact_extractor(soup)
    
    # Add metadata to each fact
    current_time = datetime.now()
    for fact in facts:
        if 'reliability_rating' not in fact:
            fact['reliability_rating'] = reliability
        if 'date_recorded' not in fact:
            fact['date_recorded'] = current_time
        if 'last_updated' not in fact:
            fact['last_updated'] = current_time
        if 'source_title' not in fact:
            fact['source_title'] = soup.title.text if soup.title else url
        if 'url_reference' not in fact:
            fact['url_reference'] = url
            
    return facts


def filter_facts_by_reliability(kg: KnowledgeGraph, 
                              min_reliability: ReliabilityRating) -> KnowledgeGraph:
    """
    Create a new knowledge graph containing only facts with at least the specified reliability.
    
    Args:
        kg: Source KnowledgeGraph instance
        min_reliability: Minimum reliability rating to include
        
    Returns:
        New KnowledgeGraph instance with filtered facts
    """
    filtered_kg = KnowledgeGraph()
    
    for node, data in kg.graph.nodes(data=True):
        if data.get('reliability_rating', ReliabilityRating.UNVERIFIED).value >= min_reliability.value:
            # Copy all attributes for the fact
            filtered_kg.add_fact(fact_id=node, **data)
            
    return filtered_kg


def merge_knowledge_graphs(kg1: KnowledgeGraph, 
                          kg2: KnowledgeGraph,
                          conflict_strategy: str = 'higher_reliability') -> KnowledgeGraph:
    """
    Merge two knowledge graphs into a new one.
    
    Args:
        kg1: First KnowledgeGraph instance
        kg2: Second KnowledgeGraph instance
        conflict_strategy: Strategy for handling conflicting facts:
                          'higher_reliability': Keep the fact with higher reliability
                          'keep_first': Keep facts from the first graph
                          'keep_second': Keep facts from the second graph
        
    Returns:
        New KnowledgeGraph instance containing merged facts
        
    Raises:
        ValueError: If an invalid conflict_strategy is provided
    """
    if conflict_strategy not in ['higher_reliability', 'keep_first', 'keep_second']:
        raise ValueError(f"Invalid conflict strategy: {conflict_strategy}")
        
    merged_kg = KnowledgeGraph()
    
    # Add all facts from the first graph
    for node, data in kg1.graph.nodes(data=True):
        merged_kg.add_fact(fact_id=node, **data)
        
    # Add facts from the second graph, handling conflicts
    for node, data in kg2.graph.nodes(data=True):
        if node not in merged_kg.graph:
            # No conflict, add the fact
            merged_kg.add_fact(fact_id=node, **data)
        else:
            # Handle conflict based on strategy
            if conflict_strategy == 'keep_first':
                continue  # Skip this fact
            elif conflict_strategy == 'keep_second':
                # Replace existing fact
                for key, value in data.items():
                    merged_kg.graph.nodes[node][key] = value
            elif conflict_strategy == 'higher_reliability':
                # Keep fact with higher reliability
                existing_reliability = merged_kg.graph.nodes[node]['reliability_rating'].value
                new_reliability = data['reliability_rating'].value
                
                if new_reliability > existing_reliability:
                    # Replace with higher reliability fact
                    for key, value in data.items():
                        merged_kg.graph.nodes[node][key] = value
                        
    return merged_kg
