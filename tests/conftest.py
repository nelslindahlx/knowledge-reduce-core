import pytest
import importlib

def check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

# Detect environments
HAS_KUZU = check_import("kuzu")
HAS_SPACY = check_import("spacy")
HAS_MATPLOTLIB = check_import("matplotlib")
HAS_NEO4J = check_import("neo4j")
HAS_MLX = check_import("mlx")
HAS_LLAMA_CPP = check_import("llama_cpp")
HAS_OLLAMA = check_import("ollama")
HAS_FASTAPI = check_import("fastapi")
HAS_GOOGLE_GENERATIVEAI = check_import("google.generativeai")

def pytest_configure(config):
    config.addinivalue_line("markers", "require_kuzu: skip if kuzu is not installed")
    config.addinivalue_line("markers", "require_spacy: skip if spacy is not installed")
    config.addinivalue_line("markers", "require_matplotlib: skip if matplotlib is not installed")
    config.addinivalue_line("markers", "require_neo4j: skip if neo4j is not installed")
    config.addinivalue_line("markers", "require_mlx: skip if mlx is not installed")
    config.addinivalue_line("markers", "require_llama_cpp: skip if llama_cpp is not installed")
    config.addinivalue_line("markers", "require_ollama: skip if ollama is not installed")
    config.addinivalue_line("markers", "require_fastapi: skip if fastapi is not installed")
    config.addinivalue_line("markers", "require_google_generativeai: skip if google-generativeai is not installed")

@pytest.fixture(autouse=True)
def skip_by_marker(request):
    """Gracefully skip tests requiring optional dependencies if they are not installed."""
    if request.node.get_closest_marker("require_kuzu") and not HAS_KUZU:
        pytest.skip("kuzu package is not installed.")
    if request.node.get_closest_marker("require_spacy") and not HAS_SPACY:
        pytest.skip("spacy package is not installed.")
    if request.node.get_closest_marker("require_matplotlib") and not HAS_MATPLOTLIB:
        pytest.skip("matplotlib package is not installed.")
    if request.node.get_closest_marker("require_neo4j") and not HAS_NEO4J:
        pytest.skip("neo4j package is not installed.")
    if request.node.get_closest_marker("require_mlx") and not HAS_MLX:
        pytest.skip("mlx package is not installed.")
    if request.node.get_closest_marker("require_llama_cpp") and not HAS_LLAMA_CPP:
        pytest.skip("llama-cpp-python package is not installed.")
    if request.node.get_closest_marker("require_ollama") and not HAS_OLLAMA:
        pytest.skip("ollama package is not installed.")
    if request.node.get_closest_marker("require_fastapi") and not HAS_FASTAPI:
        pytest.skip("fastapi package is not installed.")
    if request.node.get_closest_marker("require_google_generativeai") and not HAS_GOOGLE_GENERATIVEAI:
        pytest.skip("google-generativeai package is not installed.")
