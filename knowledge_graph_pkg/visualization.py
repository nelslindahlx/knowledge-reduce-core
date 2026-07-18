"""
Visualization utilities for knowledge graphs.

This module provides functions for visualizing knowledge graphs using
various plotting libraries and export formats.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple


def plot_knowledge_graph(graph: nx.DiGraph, 
                         title: str = "Knowledge Graph", 
                         node_size: int = 800,
                         node_color: str = "skyblue",
                         edge_color: str = "gray",
                         font_size: int = 10,
                         with_labels: bool = True,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot a knowledge graph using matplotlib and networkx.
    
    Args:
        graph: NetworkX DiGraph object to visualize
        title: Title for the plot
        node_size: Size of nodes in the visualization
        node_color: Color of nodes
        edge_color: Color of edges
        font_size: Size of label font
        with_labels: Whether to display node labels
        figsize: Figure size as (width, height) tuple
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the graph layout
    pos = nx.spring_layout(graph, seed=42)  # For reproducible layout
    
    # Draw the graph
    nx.draw(graph, pos, 
            with_labels=with_labels,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            font_size=font_size,
            ax=ax)
    
    # Set title
    plt.title(title)
    
    return fig


def export_to_gexf(graph: nx.DiGraph, filename: str) -> None:
    """
    Export a knowledge graph to GEXF format for visualization in tools like Gephi.
    
    Args:
        graph: NetworkX DiGraph object to export
        filename: Output filename (should end with .gexf)
        
    Returns:
        None
    """
    if not filename.endswith('.gexf'):
        filename += '.gexf'
    
    nx.write_gexf(graph, filename)
    

def export_to_graphml(graph: nx.DiGraph, filename: str) -> None:
    """
    Export a knowledge graph to GraphML format.
    
    Args:
        graph: NetworkX DiGraph object to export
        filename: Output filename (should end with .graphml)
        
    Returns:
        None
    """
    if not filename.endswith('.graphml'):
        filename += '.graphml'
    
    nx.write_graphml(graph, filename)


def get_graph_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate and return statistics about the knowledge graph.
    
    Args:
        graph: NetworkX DiGraph object to analyze
        
    Returns:
        Dictionary containing graph statistics
    """
    stats = {
        'node_count': graph.number_of_nodes(),
        'edge_count': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_directed': nx.is_directed(graph),
        'is_connected': nx.is_weakly_connected(graph),
        'average_clustering': nx.average_clustering(graph),
        'average_shortest_path_length': None,
    }
    
    # Calculate average shortest path length if the graph is connected
    if stats['is_connected']:
        try:
            stats['average_shortest_path_length'] = nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            # Handle case where graph has multiple components
            stats['average_shortest_path_length'] = "Graph has multiple components"
    
    return stats


def plot_reliability_distribution(graph: nx.DiGraph, 
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of reliability ratings in the knowledge graph.
    
    Args:
        graph: NetworkX DiGraph object to analyze
        figsize: Figure size as (width, height) tuple
        
    Returns:
        matplotlib Figure object
    """
    # Extract reliability ratings from all nodes
    reliability_counts = {}
    
    for node, data in graph.nodes(data=True):
        if 'reliability_rating' in data:
            rating = data['reliability_rating'].name
            reliability_counts[rating] = reliability_counts.get(rating, 0) + 1
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by reliability level
    sorted_items = sorted(reliability_counts.items(), 
                         key=lambda x: ['UNVERIFIED', 'POSSIBLY_TRUE', 'LIKELY_TRUE', 'VERIFIED'].index(x[0]))
    
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    ax.bar(labels, values, color=['red', 'orange', 'lightgreen', 'darkgreen'])
    
    # Add labels and title
    ax.set_xlabel('Reliability Rating')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Fact Reliability Ratings')
    
    # Add count labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    return fig
