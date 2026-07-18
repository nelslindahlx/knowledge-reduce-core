"""
Real-time knowledge graph with streaming data integration.

This module provides capabilities for real-time knowledge graph updates,
streaming data integration, and event-driven fact management.
"""

import time
import threading
import queue
import json
from typing import Dict, List, Any, Union, Optional, Tuple, Set, Callable
from datetime import datetime
from .core import KnowledgeGraph, ReliabilityRating

class StreamingKnowledgeGraph:
    """
    Class for real-time streaming knowledge graph operations.
    
    This class provides methods for real-time updates, event processing,
    and streaming data integration with knowledge graphs.
    
    Attributes:
        kg: The knowledge graph to enhance with streaming capabilities
        event_queue: Queue for processing incoming events
        processors: Dictionary of event processors
        running: Whether the event processing thread is running
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, auto_start: bool = True):
        """
        Initialize a StreamingKnowledgeGraph with a knowledge graph.
        
        Args:
            knowledge_graph: KnowledgeGraph instance to enhance
            auto_start: Whether to automatically start the event processing thread
        """
        self.kg = knowledge_graph
        self.event_queue = queue.Queue()
        self.processors = {}
        self.running = False
        self.processing_thread = None
        self.event_history = []
        self.max_history = 1000
        
        # Register default processors
        self._register_default_processors()
        
        # Start processing thread if auto_start is True
        if auto_start:
            self.start()
            
    def start(self) -> None:
        """
        Start the event processing thread.
        """
        if self.running:
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_events)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self) -> None:
        """
        Stop the event processing thread.
        """
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
    def add_event(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """
        Add an event to the processing queue.
        
        Args:
            event_type: Type of event
            event_data: Event data
            
        Returns:
            Event ID
        """
        event_id = f"event_{int(time.time() * 1000)}_{hash(str(event_data)) % 10000}"
        
        event = {
            'id': event_id,
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': event_data
        }
        
        self.event_queue.put(event)
        
        return event_id
        
    def register_processor(self, event_type: str, processor: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a processor for a specific event type.
        
        Args:
            event_type: Type of event to process
            processor: Function to process the event
        """
        self.processors[event_type] = processor
        
    def add_fact_from_stream(self, 
                           fact_data: Dict[str, Any], 
                           source_id: str, 
                           reliability: ReliabilityRating = ReliabilityRating.UNVERIFIED) -> str:
        """
        Add a fact from streaming data.
        
        Args:
            fact_data: Dictionary containing fact data
            source_id: Source identifier
            reliability: Reliability rating for the fact
            
        Returns:
            Event ID
        """
        # Create an event for adding a fact
        event_data = {
            'fact_data': fact_data,
            'source_id': source_id,
            'reliability': reliability.name
        }
        
        return self.add_event('add_fact', event_data)
        
    def update_fact_from_stream(self, fact_id: str, updates: Dict[str, Any]) -> str:
        """
        Update a fact from streaming data.
        
        Args:
            fact_id: ID of the fact to update
            updates: Dictionary of updates to apply
            
        Returns:
            Event ID
        """
        # Create an event for updating a fact
        event_data = {
            'fact_id': fact_id,
            'updates': updates
        }
        
        return self.add_event('update_fact', event_data)
        
    def delete_fact_from_stream(self, fact_id: str) -> str:
        """
        Delete a fact from streaming data.
        
        Args:
            fact_id: ID of the fact to delete
            
        Returns:
            Event ID
        """
        # Create an event for deleting a fact
        event_data = {
            'fact_id': fact_id
        }
        
        return self.add_event('delete_fact', event_data)
        
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the event processing history.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of processed events
        """
        return self.event_history[-limit:]
        
    def connect_to_stream(self, stream_url: str, stream_type: str = 'json') -> None:
        """
        Connect to an external data stream.
        
        This is a placeholder method. In a real implementation, this would
        establish a connection to an external streaming data source.
        
        Args:
            stream_url: URL of the stream to connect to
            stream_type: Type of stream (e.g., 'json', 'csv')
        """
        # This is a placeholder implementation
        print(f"Connected to stream at {stream_url} of type {stream_type}")
        
        # In a real implementation, this would start a thread to read from the stream
        # and add events to the queue
        
    def export_to_stream(self, stream_url: str, stream_type: str = 'json') -> None:
        """
        Export knowledge graph updates to an external stream.
        
        This is a placeholder method. In a real implementation, this would
        establish a connection to an external streaming data sink.
        
        Args:
            stream_url: URL of the stream to export to
            stream_type: Type of stream (e.g., 'json', 'csv')
        """
        # This is a placeholder implementation
        print(f"Exporting to stream at {stream_url} of type {stream_type}")
        
        # In a real implementation, this would start a thread to write to the stream
        
    def _process_events(self) -> None:
        """
        Process events from the queue.
        """
        while self.running:
            try:
                # Get an event from the queue with a timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Process the event
                self._process_event(event)
                
                # Mark the event as done
                self.event_queue.task_done()
                
            except Exception as e:
                print(f"Error processing event: {e}")
                
    def _process_event(self, event: Dict[str, Any]) -> None:
        """
        Process a single event.
        
        Args:
            event: Event to process
        """
        event_type = event['type']
        
        # Add to history
        self.event_history.append(event)
        
        # Trim history if needed
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
            
        # Check if we have a processor for this event type
        if event_type in self.processors:
            try:
                # Process the event
                self.processors[event_type](event)
            except Exception as e:
                print(f"Error in processor for event type {event_type}: {e}")
        else:
            print(f"No processor registered for event type {event_type}")
            
    def _register_default_processors(self) -> None:
        """
        Register default event processors.
        """
        self.register_processor('add_fact', self._process_add_fact)
        self.register_processor('update_fact', self._process_update_fact)
        self.register_processor('delete_fact', self._process_delete_fact)
        
    def _process_add_fact(self, event: Dict[str, Any]) -> None:
        """
        Process an add_fact event.
        
        Args:
            event: Event to process
        """
        data = event['data']
        fact_data = data['fact_data']
        source_id = data['source_id']
        reliability = getattr(ReliabilityRating, data['reliability'])
        
        # Extract required fields
        fact_id = fact_data.get('fact_id')
        if not fact_id:
            fact_id = f"stream_{int(time.time())}_{hash(str(fact_data)) % 10000}"
            fact_data['fact_id'] = fact_id
            
        # Set defaults for required fields if not present
        now = datetime.now()
        if 'date_recorded' not in fact_data:
            fact_data['date_recorded'] = now
        if 'last_updated' not in fact_data:
            fact_data['last_updated'] = now
        if 'reliability_rating' not in fact_data:
            fact_data['reliability_rating'] = reliability
        if 'source_id' not in fact_data:
            fact_data['source_id'] = source_id
            
        try:
            # Add the fact to the knowledge graph
            self.kg.add_fact(**fact_data)
        except Exception as e:
            print(f"Error adding fact from stream: {e}")
            
    def _process_update_fact(self, event: Dict[str, Any]) -> None:
        """
        Process an update_fact event.
        
        Args:
            event: Event to process
        """
        data = event['data']
        fact_id = data['fact_id']
        updates = data['updates']
        
        try:
            # Update the fact in the knowledge graph
            self.kg.update_fact(fact_id, **updates)
        except Exception as e:
            print(f"Error updating fact from stream: {e}")
            
    def _process_delete_fact(self, event: Dict[str, Any]) -> None:
        """
        Process a delete_fact event.
        
        Args:
            event: Event to process
        """
        data = event['data']
        fact_id = data['fact_id']
        
        try:
            # Check if the fact exists
            if fact_id in self.kg.graph:
                # Remove the fact from the knowledge graph
                self.kg.graph.remove_node(fact_id)
            else:
                print(f"Fact {fact_id} not found for deletion")
        except Exception as e:
            print(f"Error deleting fact from stream: {e}")
