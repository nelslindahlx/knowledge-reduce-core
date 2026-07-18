from .graph_store_base import BaseGraphStore
from .kuzu_store import KuzuStore

def get_graph_store(connection_string: str) -> BaseGraphStore:
    """Return a BaseGraphStore instance matching the connection string format.

    If the connection string starts with 'bolt://', 'neo4j://', 'neo4j+s://',
    or 'neo4j+ssc://', return a Neo4jStore.
    Otherwise, treat it as a local folder path and return a KuzuStore.
    """
    conn_lower = connection_string.lower()
    if conn_lower.startswith(("bolt://", "neo4j://", "neo4j+s://", "neo4j+ssc://")):
        from .neo4j_store import Neo4jStore
        return Neo4jStore(connection_string)
    else:
        return KuzuStore(connection_string)
