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
        from urllib.parse import urlparse, parse_qs
        
        parsed = urlparse(connection_string)
        query_params = parse_qs(parsed.query)
        db_name = query_params.get("database", [None])[0]
        
        if not db_name and parsed.path and parsed.path != "/":
            db_name = parsed.path.lstrip("/")
            
        clean_uri = f"{parsed.scheme}://{parsed.netloc}"
        user = parsed.username or "neo4j"
        password = parsed.password or "password"
        
        return Neo4jStore(clean_uri, user=user, password=password, database=db_name)
    else:
        return KuzuStore(connection_string)
