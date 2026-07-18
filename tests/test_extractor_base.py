"""
Tests for the pluggable extractor interface.

All extractors conform to the Extractor protocol: an .extract(text) method
returning a list of relation dicts with subject/predicate/object (+ types).
This lets the pipeline swap engines (SVO heuristic, spaCy, future LLM)
behind one interface and score them all on the same gold set.
"""
import importlib.util

import pytest

from knowledge_graph_pkg.extractor_base import Extractor, get_extractor
from knowledge_graph_pkg.extraction import SVOExtractor


def test_svo_conforms_to_protocol():
    ex = SVOExtractor()
    assert isinstance(ex, Extractor)
    rels = ex.extract("Robert Putnam wrote Bowling Alone.")
    assert isinstance(rels, list)
    assert all({"subject", "predicate", "object"} <= set(r) for r in rels)


def test_get_extractor_svo_default():
    ex = get_extractor("svo")
    assert isinstance(ex, Extractor)


def test_get_extractor_unknown_raises():
    with pytest.raises(ValueError):
        get_extractor("does-not-exist")


_HAS_SPACY = importlib.util.find_spec("spacy") is not None


@pytest.mark.skipif(not _HAS_SPACY, reason="spaCy not installed (optional [nlp] extra)")
def test_spacy_backend_available_when_installed():
    from knowledge_graph_pkg.spacy_extractor import SpacyExtractor
    ex = SpacyExtractor()
    assert isinstance(ex, Extractor)
    rels = ex.extract("Robert Putnam wrote Bowling Alone.")
    assert isinstance(rels, list)


def test_get_extractor_spacy_without_install_gives_clear_error():
    if _HAS_SPACY:
        pytest.skip("spaCy is installed; the error path isn't exercised")
    with pytest.raises(ImportError, match="nlp"):
        get_extractor("spacy")
