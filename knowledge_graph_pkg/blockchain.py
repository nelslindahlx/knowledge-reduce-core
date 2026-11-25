"""
Distributed knowledge graph with blockchain integration.

This module provides capabilities for distributed knowledge graphs with
blockchain-based verification, consensus mechanisms, and immutable fact history.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from .core import KnowledgeGraph, ReliabilityRating


def _normalize_tags_str(val: Any) -> str:
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x).strip() for x in val if str(x).strip())
    if isinstance(val, str):
        return ", ".join(x.strip() for x in val.split(",") if x.strip())
    return str(val or "")


def _json_safe(val: Any) -> Any:
    if isinstance(val, ReliabilityRating):
        return val.name
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, dict):
        return {k: _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(v) for v in val]
    return val


def _parse_reliability(val: Any) -> ReliabilityRating:
    if isinstance(val, ReliabilityRating):
        return val
    if isinstance(val, str):
        member = ReliabilityRating.__members__.get(val.strip().upper())
        if member is not None:
            return member
    raise ValueError(f"Invalid reliability rating: {val!r}")


class BlockchainKnowledgeGraph:
    """
    Class for blockchain-based distributed knowledge graphs.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, difficulty: int = 2):
        """
        Initialize a BlockchainKnowledgeGraph with a knowledge graph.
        """
        self.kg = knowledge_graph
        self.difficulty = max(1, int(difficulty))
        self.chain = []
        self.pending_transactions = []
        
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
        """
        rel_enum = _parse_reliability(reliability_rating)
        tags_list = list(tags) if isinstance(tags, (list, tuple)) else [
            t.strip() for t in str(tags).split(",") if t.strip()
        ]
        extra = dict(kwargs)
        now = datetime.now().isoformat()

        kg_fact_data = {
            'fact_statement': fact_statement,
            'category': category,
            'tags': tags_list,
            'reliability_rating': rel_enum,
            'source_id': source_id,
            **extra,
        }
        kg_fact_data.setdefault('date_recorded', now)
        kg_fact_data.setdefault('last_updated', now)
        kg_fact_data.setdefault('related_facts', [])
        kg_fact_data.setdefault('contextual_notes', "")
        kg_fact_data.setdefault('access_level', "public")
        kg_fact_data.setdefault('usage_count', 0)
        kg_fact_data.setdefault('publication_date', now)
        for field in ('source_title', 'author_creator', 'url_reference'):
            kg_fact_data.setdefault(field, "")

        # Validate and apply to KG BEFORE recording transaction so failed adds never corrupt ledger.
        self.kg.add_fact(fact_id=fact_id, **kg_fact_data)

        tx_payload = _json_safe({
            'fact_statement': fact_statement,
            'category': category,
            'tags': tags_list,
            'reliability_rating': rel_enum.name,
            'source_id': source_id,
            **extra,
        })
        transaction = {
            'type': 'add_fact',
            'timestamp': datetime.now().isoformat(),
            'fact_id': fact_id,
            'fact_data': tx_payload,
        }
        return self._add_transaction(transaction)
        
    def update_fact(self, fact_id: str, **updates) -> str:
        """
        Update a fact with blockchain verification.
        """
        kg_updates = dict(updates)
        if 'reliability_rating' in kg_updates:
            kg_updates['reliability_rating'] = _parse_reliability(
                kg_updates['reliability_rating']
            )

        # Apply to KG first so invalid updates fail cleanly before writing ledger.
        self.kg.update_fact(fact_id, **kg_updates)

        transaction = {
            'type': 'update_fact',
            'timestamp': datetime.now().isoformat(),
            'fact_id': fact_id,
            'updates': _json_safe(updates),
        }
        return self._add_transaction(transaction)
        
    def verify_fact(self, fact_id: str) -> Dict[str, Any]:
        """
        Verify a fact using the blockchain.
        """
        try:
            fact = self.kg.get_fact(fact_id)
        except Exception as e:
            return {
                'verified': False,
                'error': f"Fact not found: {e}"
            }
            
        transactions = self._get_fact_transactions(fact_id)
        
        if not transactions:
            return {
                'verified': False,
                'error': "No blockchain record found for this fact"
            }
            
        expected_state = self._compute_expected_state(fact_id, transactions)
        
        matches = True
        mismatches = []
        
        for key, value in expected_state.items():
            if key in fact:
                if key == 'tags':
                    if _normalize_tags_str(fact[key]) != _normalize_tags_str(value):
                        matches = False
                        mismatches.append(key)
                elif key == 'reliability_rating':
                    actual_name = (
                        fact[key].name
                        if isinstance(fact[key], ReliabilityRating)
                        else str(fact[key]).strip().upper()
                    )
                    expected_name = (
                        value.name
                        if isinstance(value, ReliabilityRating)
                        else str(value).strip().upper()
                    )
                    if actual_name != expected_name:
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
        
    def mine_block(self) -> Optional[Dict[str, Any]]:
        """
        Mine a new block with pending transactions.
        """
        if not self.pending_transactions:
            return None
            
        previous_block = self.chain[-1]
        
        block = {
            'index': len(self.chain),
            'timestamp': datetime.now().isoformat(),
            'transactions': list(self.pending_transactions),
            'previous_hash': previous_block['hash'],
            'nonce': 0
        }
        
        block = self._proof_of_work(block)
        self.chain.append(block)
        self.pending_transactions = []
        
        return block
        
    def get_fact_history(self, fact_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete history of a fact from the blockchain.
        """
        return self._get_fact_transactions(fact_id)
        
    def verify_chain(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire blockchain.
        """
        if not self.chain:
            return {'valid': False, 'error': "Chain is empty"}

        genesis = self.chain[0]
        if genesis.get('previous_hash') != '0' or not self._is_valid_hash(genesis, check_difficulty=False):
            return {'valid': False, 'error': "Block 0 (genesis) has invalid hash"}

        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            if current_block.get('previous_hash') != previous_block.get('hash'):
                return {
                    'valid': False,
                    'error': f"Block {i} has invalid previous hash"
                }
                
            if not self._is_valid_hash(current_block, check_difficulty=True):
                return {
                    'valid': False,
                    'error': f"Block {i} has invalid hash"
                }
                
        return {
            'valid': True,
            'blocks': len(self.chain),
            'transactions': sum(len(block.get('transactions', [])) for block in self.chain)
        }
        
    def export_chain(self, filename: str) -> None:
        """
        Export the blockchain to a file atomically.
        """
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{filename}.tmp.{os.getpid()}"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.chain, f, indent=2)
        os.replace(tmp_path, filename)
            
    def import_chain(self, filename: str) -> int:
        """
        Import a blockchain from a file, preserving the current chain on failure.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            chain = json.load(f)

        if not isinstance(chain, list):
            raise ValueError("Invalid blockchain: expected list of blocks")
            
        previous_chain = list(self.chain)
        self.chain = chain
        try:
            verification = self.verify_chain()
            if not verification['valid']:
                raise ValueError(f"Invalid blockchain: {verification['error']}")
            self._rebuild_knowledge_graph()
        except Exception:
            self.chain = previous_chain
            raise
        
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
        """
        target = '0' * self.difficulty
        
        while True:
            block_hash = self._calculate_hash(block)
            if block_hash.startswith(target):
                block['hash'] = block_hash
                return block
            block['nonce'] += 1
            
    def _is_valid_hash(self, block: Dict[str, Any], check_difficulty: bool = True) -> bool:
        """
        Check if a block has a valid hash (and difficulty prefix when mined).
        """
        expected_hash = self._calculate_hash(block)
        actual_hash = block.get('hash')
        if actual_hash != expected_hash:
            return False
        if check_difficulty and not actual_hash.startswith('0' * self.difficulty):
            return False
        return True
        
    def _get_fact_transactions(self, fact_id: str) -> List[Dict[str, Any]]:
        """
        Get all transactions related to a fact.
        """
        transactions = []
        
        for block in self.chain:
            for tx in block['transactions']:
                if tx['type'] in ['add_fact', 'update_fact'] and tx['fact_id'] == fact_id:
                    transactions.append(tx)
                    
        for tx in self.pending_transactions:
            if tx['type'] in ['add_fact', 'update_fact'] and tx['fact_id'] == fact_id:
                transactions.append(tx)
                
        return transactions
        
    def _compute_expected_state(self, fact_id: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute the expected state of a fact based on its transaction history.
        """
        state = {}
        
        for tx in transactions:
            if tx['type'] == 'add_fact':
                state = tx['fact_data'].copy()
            elif tx['type'] == 'update_fact':
                for key, value in tx['updates'].items():
                    state[key] = value
                    
        return state
        
    def _rebuild_knowledge_graph(self) -> None:
        """
        Rebuild the knowledge graph from the blockchain.
        """
        self.kg.graph.clear()
        
        for block in self.chain:
            for tx in block['transactions']:
                if tx['type'] == 'add_fact':
                    fact_id = tx['fact_id']
                    fact_data = tx['fact_data'].copy()
                    
                    if 'reliability_rating' in fact_data:
                        fact_data['reliability_rating'] = _parse_reliability(
                            fact_data['reliability_rating']
                        )
                        
                    now = datetime.now().isoformat()
                    fact_data.setdefault('date_recorded', now)
                    fact_data.setdefault('last_updated', now)
                    fact_data.setdefault('related_facts', [])
                    fact_data.setdefault('contextual_notes', "")
                    fact_data.setdefault('access_level', "public")
                    fact_data.setdefault('usage_count', 0)
                    fact_data.setdefault('publication_date', now)
                        
                    for field in ('source_title', 'author_creator', 'url_reference'):
                        fact_data.setdefault(field, "")
                            
                    try:
                        self.kg.add_fact(fact_id=fact_id, **fact_data)
                    except Exception as e:
                        print(f"Error adding fact during rebuild: {e}")
                        
                elif tx['type'] == 'update_fact':
                    fact_id = tx['fact_id']
                    updates = tx['updates'].copy()
                    
                    if 'reliability_rating' in updates:
                        updates['reliability_rating'] = _parse_reliability(
                            updates['reliability_rating']
                        )
                        
                    try:
                        if fact_id in self.kg.graph:
                            self.kg.update_fact(fact_id, **updates)
                    except Exception as e:
                        print(f"Error updating fact during rebuild: {e}")
