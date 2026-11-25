from urllib.parse import parse_qs, unquote, urlparse

from .graph_store_base import BaseGraphStore
from .kuzu_store import KuzuStore

_NEO4J_SCHEMES = (
    "bolt://",
    "bolt+s://",
    "bolt+ssc://",
    "neo4j://",
    "neo4j+s://",
    "neo4j+ssc://",
)


def is_neo4j_uri(connection_string: str) -> bool:
    """Return True if ``connection_string`` uses a Neo4j/Bolt URI scheme."""
    return connection_string.lower().startswith(_NEO4J_SCHEMES)


def get_graph_store(connection_string: str) -> BaseGraphStore:
    """Return a BaseGraphStore instance matching the connection string format.

    If the connection string starts with a Neo4j/Bolt URI scheme ('bolt://',
    'bolt+s://', 'bolt+ssc://', 'neo4j://', 'neo4j+s://', or 'neo4j+ssc://'),
    return a Neo4jStore. Otherwise, treat it as a local folder path and return
    a KuzuStore.
    """
    if is_neo4j_uri(connection_string):
        from .neo4j_store import Neo4jStore

        parsed = urlparse(connection_string)
        query_params = parse_qs(parsed.query)
        db_name = query_params.get("database", [None])[0]

        if not db_name and parsed.path and parsed.path != "/":
            db_name = parsed.path.lstrip("/")

        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        clean_uri = f"{parsed.scheme}://{host}"
        if parsed.port is not None:
            clean_uri = f"{clean_uri}:{parsed.port}"

        user = unquote(parsed.username) if parsed.username is not None else None
        password = unquote(parsed.password) if parsed.password is not None else None

        return Neo4jStore(clean_uri, user=user, password=password, database=db_name)
    return KuzuStore(connection_string)
