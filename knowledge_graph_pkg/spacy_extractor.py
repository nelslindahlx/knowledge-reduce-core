"""
spaCy-based relation extractor (optional ``[nlp]`` extra).

Uses spaCy's dependency parse to find subject-verb-object structures,
which handles nested clauses, possessives, and varied phrasing far better
than the pure-Python heuristic. This module imports spaCy lazily, so the
rest of the package works without it; install with::

    pip install knowledgereduce[nlp]
    python -m spacy download en_core_web_sm

The output dicts match :class:`SVOExtractor` exactly
(subject/predicate/object/subject_type/object_type/text) so the two are
interchangeable behind the :class:`Extractor` interface.
"""

from typing import Any, Dict, List, Optional

_DEFAULT_MODEL = "en_core_web_sm"


class SpacyExtractor:
    """Dependency-parse relation extractor backed by spaCy."""

    def __init__(self, model: str = _DEFAULT_MODEL):
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "SpacyExtractor requires the [nlp] extra: "
                "pip install knowledgereduce[nlp]"
            ) from exc
        try:
            self._nlp = spacy.load(model, disable=["ner", "lemmatizer"])
        except OSError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                f"spaCy model '{model}' not found. Install it with: "
                f"python -m spacy download {model}"
            ) from exc

    # ------------------------------------------------------------------ #
    @staticmethod
    def _coarse_type(span_text: str) -> str:
        return "ENTITY" if span_text[:1].isupper() else "CONCEPT"

    @staticmethod
    def _subtree_text(token) -> str:
        """Return the text of a token's subtree, trimmed of leading articles."""
        words = [t.text for t in token.subtree]
        text = " ".join(words).strip()
        for art in ("the ", "a ", "an ", "The ", "A ", "An "):
            if text.startswith(art):
                text = text[len(art):]
                break
        return text.strip(" ,.;:")

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract subject-verb-object triples using the dependency parse."""
        relations: List[Dict[str, Any]] = []
        doc = self._nlp(text)

        for sent in doc.sents:
            for token in sent:
                if token.pos_ not in ("VERB", "AUX"):
                    continue

                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                if not subjects:
                    continue

                # Objects: direct objects, attributes, and prepositional objects.
                objects = []
                for child in token.children:
                    if child.dep_ in ("dobj", "attr", "oprd", "dative"):
                        objects.append(child)
                    elif child.dep_ == "prep":
                        objects.extend(
                            gc for gc in child.children if gc.dep_ == "pobj"
                        )
                if not objects:
                    continue

                predicate = token.lemma_.lower() if token.pos_ == "VERB" else token.text.lower()

                for subj in subjects:
                    subj_text = self._subtree_minimal(subj)
                    for obj in objects:
                        obj_text = self._subtree_text(obj)
                        if not subj_text or not obj_text:
                            continue
                        relations.append({
                            "subject": subj_text,
                            "subject_type": self._coarse_type(subj_text),
                            "predicate": predicate.replace(" ", "_"),
                            "object": obj_text,
                            "object_type": self._coarse_type(obj_text),
                            "text": sent.text.strip(),
                        })
        return relations

    @staticmethod
    def _subtree_minimal(token) -> str:
        """Compact subject: the token plus contiguous proper-noun/compound parts."""
        parts = [t for t in token.subtree if t.dep_ in ("compound", "flat", "poss") or t == token]
        parts = sorted(set(parts), key=lambda t: t.i)
        text = " ".join(t.text for t in parts).strip(" ,.;:")
        return text or token.text
