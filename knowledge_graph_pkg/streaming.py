"""
Real-time knowledge graph with streaming data integration.

This module provides capabilities for real-time knowledge graph updates,
streaming data integration, and event-driven fact management.
"""

import time
import threading
import queue
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from .core import KnowledgeGraph, ReliabilityRating


def _parse_reliability(val: Any, default: ReliabilityRating = ReliabilityRating.UNVERIFIED) -> ReliabilityRating:
    if isinstance(val, ReliabilityRating):
        return val
    if isinstance(val, str):
        member = ReliabilityRating.__members__.get(val.strip().upper())
        if member is not None:
            return member
    return default


class StreamingKnowledgeGraph:
    """
    Class for real-time streaming knowledge graph operations.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, auto_start: bool = True):
        """
        Initialize a StreamingKnowledgeGraph with a knowledge graph.
        """
        self.kg = knowledge_graph
        self.event_queue = queue.Queue()
        self.processors = {}
        self.running = False
        self.processing_thread = None
        self.event_history = []
        self.max_history = 1000
        self._lock = threading.RLock()
        
        self._register_default_processors()
        
        if auto_start:
            self.start()
            
    def start(self) -> None:
        """
        Start the event processing thread.
        """
        with self._lock:
            if self.running:
                return
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
            self.processing_thread.start()
        
    def stop(self) -> None:
        """
        Stop the event processing thread.
        """
        with self._lock:
            self.running = False
            thread = self.processing_thread
        if thread:
            thread.join(timeout=2.0)
            
    def add_event(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """
        Add an event to the processing queue.
        """
        event_id = f"event_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        event = {
            'id': event_id,
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': dict(event_data),
        }
        
        self.event_queue.put(event)
        return event_id
        
    def register_processor(self, event_type: str, processor: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a processor for a specific event type.
        """
        with self._lock:
            self.processors[event_type] = processor
        
    def add_fact_from_stream(self, 
                           fact_data: Dict[str, Any], 
                           source_id: str, 
                           reliability: ReliabilityRating = ReliabilityRating.UNVERIFIED) -> str:
        """
        Add a fact from streaming data.
        """
        rel_enum = _parse_reliability(reliability)
        event_data = {
            'fact_data': dict(fact_data),
            'source_id': source_id,
            'reliability': rel_enum.name,
        }
        return self.add_event('add_fact', event_data)
        
    def update_fact_from_stream(self, fact_id: str, updates: Dict[str, Any]) -> str:
        """
        Update a fact from streaming data.
        """
        event_data = {
            'fact_id': fact_id,
            'updates': dict(updates),
        }
        return self.add_event('update_fact', event_data)
        
    def delete_fact_from_stream(self, fact_id: str) -> str:
        """
        Delete a fact from streaming data.
        """
        event_data = {'fact_id': fact_id}
        return self.add_event('delete_fact', event_data)
        
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the event processing history.
        """
        if limit <= 0:
            return []
        with self._lock:
            return list(self.event_history[-limit:])
        
    def connect_to_stream(self, stream_url: str, stream_type: str = 'json') -> None:
        """
        Connect to an external data stream.
        """
        print(f"Connected to stream at {stream_url} of type {stream_type}")
        
    def export_to_stream(self, stream_url: str, stream_type: str = 'json') -> None:
        """
        Export knowledge graph updates to an external stream.
        """
        print(f"Exporting to stream at {stream_url} of type {stream_type}")
        
    def _process_events(self) -> None:
        """
        Process events from the queue.
        """
        while self.running:
            try:
                event = self.event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._process_event(event)
            except Exception as e:
                print(f"Error processing event: {e}")
            finally:
                self.event_queue.task_done()
                
    def _process_event(self, event: Dict[str, Any]) -> None:
        """
        Process a single event.
        """
        event_type = event['type']
        
        with self._lock:
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
            processor = self.processors.get(event_type)
            
        if processor is not None:
            try:
                processor(event)
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
        Process an add_fact event without mutating caller dicts.
        """
        data = event['data']
        fact_data = dict(data['fact_data'])
        fact_data.pop('quality_score', None)
        source_id = data['source_id']
        default_rel = _parse_reliability(data.get('reliability'))
        
        fact_id = fact_data.pop('fact_id', None)
        if not fact_id:
            fact_id = f"stream_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            
        now = datetime.now().isoformat()
        raw_tags = fact_data.get('tags', [])
        if isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(',') if t.strip()]
        elif isinstance(raw_tags, (list, tuple)):
            tags = list(raw_tags)
        else:
            tags = []

        rel = _parse_reliability(fact_data.get('reliability_rating'), default=default_rel)

        try:
            with self._lock:
                self.kg.add_fact(
                    fact_id=fact_id,
                    fact_statement=str(fact_data.get('fact_statement', '')),
                    category=str(fact_data.get('category') or 'General'),
                    tags=tags,
                    date_recorded=fact_data.get('date_recorded') or now,
                    last_updated=fact_data.get('last_updated') or now,
                    reliability_rating=rel,
                    source_id=str(fact_data.get('source_id') or source_id),
                    source_title=str(fact_data.get('source_title') or ''),
                    author_creator=str(fact_data.get('author_creator') or ''),
                    publication_date=fact_data.get('publication_date') or now,
                    url_reference=str(fact_data.get('url_reference') or ''),
                    related_facts=list(fact_data.get('related_facts') or []),
                    contextual_notes=str(fact_data.get('contextual_notes') or ''),
                    access_level=str(fact_data.get('access_level') or 'public'),
                    usage_count=int(fact_data.get('usage_count') or 0),
                )
        except Exception as e:
            print(f"Error adding fact from stream: {e}")
            
    def _process_update_fact(self, event: Dict[str, Any]) -> None:
        """
        Process an update_fact event.
        """
        data = event['data']
        fact_id = data['fact_id']
        updates = dict(data['updates'])
        if 'reliability_rating' in updates and isinstance(updates['reliability_rating'], str):
            updates['reliability_rating'] = ReliabilityRating.__members__.get(
                updates['reliability_rating'].strip().upper(),
                updates['reliability_rating'],
            )
        
        try:
            with self._lock:
                self.kg.update_fact(fact_id, **updates)
        except Exception as e:
            print(f"Error updating fact from stream: {e}")
            
    def _process_delete_fact(self, event: Dict[str, Any]) -> None:
        """
        Process a delete_fact event.
        """
        data = event['data']
        fact_id = data['fact_id']
        
        try:
            with self._lock:
                if fact_id in self.kg.graph:
                    self.kg.graph.remove_node(fact_id)
                else:
                    print(f"Fact {fact_id} not found for deletion")
        except Exception as e:
            print(f"Error deleting fact from stream: {e}")
