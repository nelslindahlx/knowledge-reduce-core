"""
Pluggable extractor interface.

An :class:`Extractor` is anything with an ``extract(text) -> list[dict]``
method returning relation triples (subject/predicate/object plus coarse
types). This lets the pipeline swap extraction engines behind one
interface and score them all on the same gold set:

* ``svo``   -- the default pure-Python heuristic (:class:`SVOExtractor`).
* ``spacy`` -- dependency-parse backend, optional ``[nlp]`` extra.

Use :func:`get_extractor` to obtain an engine by name.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Extractor(Protocol):
    """Structural type for relation extractors."""

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Return a list of relation dicts for the given text."""
        ...


def get_extractor(name: str = "svo", **kwargs: Any) -> Extractor:
    """Return an extractor engine by name.

    Args:
        name: ``"svo"`` (default, pure-Python) or ``"spacy"`` (optional
            ``[nlp]`` extra).
        **kwargs: Forwarded to the engine constructor.

    Raises:
        ValueError: If ``name`` is not a known engine.
        ImportError: If ``spacy`` is requested but the extra isn't installed.
    """
    key = (name or "svo").lower()
    if key == "svo":
        from .extraction import SVOExtractor
        return SVOExtractor(**kwargs)
    if key == "spacy":
        try:
            from .spacy_extractor import SpacyExtractor
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "The 'spacy' engine requires the optional [nlp] extra: "
                "pip install knowledgereduce[nlp] && python -m spacy download en_core_web_sm"
            ) from exc
        return SpacyExtractor(**kwargs)
    raise ValueError(f"Unknown extractor engine: {name!r}. Choose 'svo' or 'spacy'.")
