"""
Distributed knowledge graph with blockchain integration.

This module provides capabilities for distributed knowledge graphs with
blockchain-based verification, consensus mechanisms, and immutable fact history.
"""

import hashlib
import time
import json
from typing import Dict, List, Any, Union, Optional, Tuple, Set, Callable
from datetime import datetime
from .core import KnowledgeGraph, ReliabilityRating

class BlockchainKnowledgeGraph:
    """
    Class for blockchain-based distributed knowledge graphs.
    
    This class provides methods for creating distributed knowledge graphs
    with blockchain-based verification and immutable history.
    
    Attributes:
        kg: The knowledge graph to enhance with blockchain capabilities
        chain: The blockchain containing fact history
        pending_transactions: Transactions waiting to be added to the blockchain
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Initialize a BlockchainKnowledgeGraph with a knowledge graph.
        
        Args:
            knowledge_graph: KnowledgeGraph instance to enhance
        """
        self.kg = knowledge_graph
        self.chain = []
        self.pending_transactions = []
        
        # Create genesis block
        self._create_genesis_block()
        
    def add_fact(self, 
                fact_id: str, 
                fact_statement: str, 
                category: str, 
                tags: List[str], 
                reliability_rating: ReliabilityRating, 
                source_id: str, 
                **kwargs) -> str:
        """
        Add a new fact with blockchain verification.
        
        Args:
            fact_id: Unique identifier for the fact
            fact_statement: The actual fact statement
            category: Category of the fact
            tags: List of tags associated with the fact
            reliability_rating: ReliabilityRating enum value
            source_id: Identifier for the source
            **kwargs: Additional fact attributes
            
        Returns:
            Transaction hash
        """
        # Create transaction data
        transaction = {
            'type': 'add_fact',
            'timestamp': datetime.now().isoformat(),
            'fact_id': fact_id,
            'fact_data': {
                'fact_statement': fact_statement,
                'category': category,
                'tags': tags,
                'reliability_rating': reliability_rating.name,
                'source_id': source_id,
                **kwargs
            }
        }
        
        # Add transaction to pending transactions
        tx_hash = self._add_transaction(transaction)
        
        # Add the fact to the knowledge graph
        fact_data = transaction['fact_data'].copy()
        fact_data['reliability_rating'] = reliability_rating
        
        # Set defaults for required fields
        now = datetime.now()
        if 'date_recorded' not in fact_data:
            fact_data['date_recorded'] = now
        if 'last_updated' not in fact_data:
            fact_data['last_updated'] = now
        if 'related_facts' not in fact_data:
            fact_data['related_facts'] = []
        if 'contextual_notes' not in fact_data:
            fact_data['contextual_notes'] = ""
        if 'access_level' not in fact_data:
            fact_data['access_level'] = "public"
        if 'usage_count' not in fact_data:
            fact_data['usage_count'] = 0
            
        # Add required fields that might be missing
        for field in ['source_title', 'author_creator', 'publication_date', 'url_reference']:
            if field not in fact_data:
                fact_data[field] = ""
                
        try:
            self.kg.add_fact(fact_id=fact_id, **fact_data)
        except Exception as e:
            print(f"Error adding fact to knowledge graph: {e}")
            
        return tx_hash
        
    def update_fact(self, fact_id: str, **updates) -> str:
        """
        Update a fact with blockchain verification.
        
        Args:
            fact_id: ID of the fact to update
            **updates: Attributes to update
            
        Returns:
            Transaction hash
        """
        # Create transaction data
        transaction = {
            'type': 'update_fact',
            'timestamp': datetime.now().isoformat(),
            'fact_id': fact_id,
            'updates': updates
        }
        
        # Add transaction to pending transactions
        tx_hash = self._add_transaction(transaction)
        
        # Update the fact in the knowledge graph
        try:
            # Convert reliability_rating from string to enum if present
            if 'reliability_rating' in updates and isinstance(updates['reliability_rating'], str):
                updates['reliability_rating'] = getattr(ReliabilityRating, updates['reliability_rating'])
                
            self.kg.update_fact(fact_id, **updates)
        except Exception as e:
            print(f"Error updating fact in knowledge graph: {e}")
            
        return tx_hash
        
    def verify_fact(self, fact_id: str) -> Dict[str, Any]:
        """
        Verify a fact using the blockchain.
        
        Args:
            fact_id: ID of the fact to verify
            
        Returns:
            Verification result
        """
        # Get the fact from the knowledge graph
        try:
            fact = self.kg.get_fact(fact_id)
        except Exception as e:
            return {
                'verified': False,
                'error': f"Fact not found: {e}"
            }
            
        # Find all transactions related to this fact
        transactions = self._get_fact_transactions(fact_id)
        
        if not transactions:
            return {
                'verified': False,
                'error': "No blockchain record found for this fact"
            }
            
        # Verify the current state matches the blockchain history
        expected_state = self._compute_expected_state(fact_id, transactions)
        
        # Compare with actual state
        matches = True
        mismatches = []
        
        for key, value in expected_state.items():
            if key in fact:
                if isinstance(fact[key], ReliabilityRating) and isinstance(value, str):
                    # Compare enum with string
                    if fact[key].name != value:
                        matches = False
                        mismatches.append(key)
                elif str(fact[key]) != str(value):
                    matches = False
                    mismatches.append(key)
                    
        return {
            'verified': matches,
            'transactions': len(transactions),
            'first_recorded': transactions[0]['timestamp'],
            'last_updated': transactions[-1]['timestamp'],
            'mismatches': mismatches if not matches else []
        }
        
    def mine_block(self) -> Dict[str, Any]:
        """
        Mine a new block with pending transactions.
        
        Returns:
            The newly mined block
        """
        if not self.pending_transactions:
            return None
            
        # Get the previous block
        previous_block = self.chain[-1]
        
        # Create a new block
        block = {
            'index': len(self.chain),
            'timestamp': datetime.now().isoformat(),
            'transactions': self.pending_transactions,
            'previous_hash': previous_block['hash'],
            'nonce': 0
        }
        
        # Mine the block (find a valid hash)
        block = self._proof_of_work(block)
        
        # Add the block to the chain
        self.chain.append(block)
        
        # Clear pending transactions
        self.pending_transactions = []
        
        return block
        
    def get_fact_history(self, fact_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete history of a fact from the blockchain.
        
        Args:
            fact_id: ID of the fact to get history for
            
        Returns:
            List of transactions related to the fact
        """
        return self._get_fact_transactions(fact_id)
        
    def verify_chain(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire blockchain.
        
        Returns:
            Verification result
        """
        # Check each block in the chain
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if the previous hash matches
            if current_block['previous_hash'] != previous_block['hash']:
                return {
                    'valid': False,
                    'error': f"Block {i} has invalid previous hash"
                }
                
            # Check if the block's hash is valid
            if not self._is_valid_hash(current_block):
                return {
                    'valid': False,
                    'error': f"Block {i} has invalid hash"
                }
                
        return {
            'valid': True,
            'blocks': len(self.chain),
            'transactions': sum(len(block['transactions']) for block in self.chain)
        }
        
    def export_chain(self, filename: str) -> None:
        """
        Export the blockchain to a file.
        
        Args:
            filename: Path to save the blockchain
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.chain, f, indent=2)
            
    def import_chain(self, filename: str) -> int:
        """
        Import a blockchain from a file.
        
        Args:
            filename: Path to load the blockchain from
            
        Returns:
            Number of blocks imported
        """
        with open(filename, 'r', encoding='utf-8') as f:
            chain = json.load(f)
            
        # Verify the chain
        self.chain = chain
        verification = self.verify_chain()
        
        if not verification['valid']:
            raise ValueError(f"Invalid blockchain: {verification['error']}")
            
        # Rebuild the knowledge graph from the chain
        self._rebuild_knowledge_graph()
        
        return len(chain)
        
    def _create_genesis_block(self) -> None:
        """
        Create the genesis block for the blockchain.
        """
        genesis_block = {
            'index': 0,
            'timestamp': datetime.now().isoformat(),
            'transactions': [],
            'previous_hash': '0',
            'nonce': 0
        }
        
        # Calculate the hash for the genesis block
        genesis_block['hash'] = self._calculate_hash(genesis_block)
        
        # Add the genesis block to the chain
        self.chain.append(genesis_block)
        
    def _add_transaction(self, transaction: Dict[str, Any]) -> str:
        """
        Add a transaction to pending transactions.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Transaction hash
        """
        # Calculate transaction hash
        tx_data = json.dumps(transaction, sort_keys=True)
        tx_hash = hashlib.sha256(tx_data.encode()).hexdigest()
        
        # Add hash to transaction
        transaction['hash'] = tx_hash
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        
        # If we have enough pending transactions, mine a block
        if len(self.pending_transactions) >= 10:
            self.mine_block()
            
        return tx_hash
        
    def _calculate_hash(self, block: Dict[str, Any]) -> str:
        """
        Calculate the hash of a block.
        
        Args:
            block: Block to calculate hash for
            
        Returns:
            Block hash
        """
        # Create a copy of the block without the hash field
        block_copy = block.copy()
        if 'hash' in block_copy:
            del block_copy['hash']
            
        # Convert the block to a string and calculate the hash
        block_string = json.dumps(block_copy, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
        
    def _proof_of_work(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform proof of work to find a valid hash.
        
        Args:
            block: Block to mine
            
        Returns:
            Mined block with valid hash
        """
        # Define the difficulty (number of leading zeros required)
        difficulty = 2
        target = '0' * difficulty
        
        while True:
            # Calculate the hash
            block_hash = self._calculate_hash(block)
            
            # Check if the hash meets the difficulty requirement
            if block_hash.startswith(target):
                block['hash'] = block_hash
                return block
                
            # Increment the nonce and try again
            block['nonce'] += 1
            
    def _is_valid_hash(self, block: Dict[str, Any]) -> bool:
        """
        Check if a block has a valid hash.
        
        Args:
            block: Block to check
            
        Returns:
            True if the hash is valid, False otherwise
        """
        # Calculate the expected hash
        expected_hash = self._calculate_hash(block)
        
        # Compare with the block's hash
        return block['hash'] == expected_hash
        
    def _get_fact_transactions(self, fact_id: str) -> List[Dict[str, Any]]:
        """
        Get all transactions related to a fact.
        
        Args:
            fact_id: ID of the fact
            
        Returns:
            List of transactions
        """
        transactions = []
        
        # Search all blocks for transactions related to this fact
        for block in self.chain:
            for tx in block['transactions']:
                if tx['type'] in ['add_fact', 'update_fact'] and tx['fact_id'] == fact_id:
                    transactions.append(tx)
                    
        # Also check pending transactions
        for tx in self.pending_transactions:
            if tx['type'] in ['add_fact', 'update_fact'] and tx['fact_id'] == fact_id:
                transactions.append(tx)
                
        return transactions
        
    def _compute_expected_state(self, fact_id: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute the expected state of a fact based on its transaction history.
        
        Args:
            fact_id: ID of the fact
            transactions: List of transactions related to the fact
            
        Returns:
            Expected state of the fact
        """
        state = {}
        
        for tx in transactions:
            if tx['type'] == 'add_fact':
                # Initial state from add_fact transaction
                state = tx['fact_data'].copy()
            elif tx['type'] == 'update_fact':
                # Apply updates
                for key, value in tx['updates'].items():
                    state[key] = value
                    
        return state
        
    def _rebuild_knowledge_graph(self) -> None:
        """
        Rebuild the knowledge graph from the blockchain.
        """
        # Clear the existing knowledge graph
        self.kg.graph.clear()
        
        # Process all transactions in the chain
        for block in self.chain:
            for tx in block['transactions']:
                if tx['type'] == 'add_fact':
                    # Add fact to knowledge graph
                    fact_id = tx['fact_id']
                    fact_data = tx['fact_data'].copy()
                    
                    # Convert reliability_rating from string to enum if present
                    if 'reliability_rating' in fact_data and isinstance(fact_data['reliability_rating'], str):
                        fact_data['reliability_rating'] = getattr(ReliabilityRating, fact_data['reliability_rating'])
                        
                    # Set defaults for required fields
                    now = datetime.now()
                    if 'date_recorded' not in fact_data:
                        fact_data['date_recorded'] = now
                    if 'last_updated' not in fact_data:
                        fact_data['last_updated'] = now
                    if 'related_facts' not in fact_data:
                        fact_data['related_facts'] = []
                    if 'contextual_notes' not in fact_data:
                        fact_data['contextual_notes'] = ""
                    if 'access_level' not in fact_data:
                        fact_data['access_level'] = "public"
                    if 'usage_count' not in fact_data:
                        fact_data['usage_count'] = 0
                        
                    # Add required fields that might be missing
                    for field in ['source_title', 'author_creator', 'publication_date', 'url_reference']:
                        if field not in fact_data:
                            fact_data[field] = ""
                            
                    try:
                        self.kg.add_fact(fact_id=fact_id, **fact_data)
                    except Exception as e:
                        print(f"Error adding fact during rebuild: {e}")
                        
                elif tx['type'] == 'update_fact':
                    # Update fact in knowledge graph
                    fact_id = tx['fact_id']
                    updates = tx['updates'].copy()
                    
                    # Convert reliability_rating from string to enum if present
                    if 'reliability_rating' in updates and isinstance(updates['reliability_rating'], str):
                        updates['reliability_rating'] = getattr(ReliabilityRating, updates['reliability_rating'])
                        
                    try:
                        if fact_id in self.kg.graph:
                            self.kg.update_fact(fact_id, **updates)
                    except Exception as e:
                        print(f"Error updating fact during rebuild: {e}")
