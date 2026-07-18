"""
Fact quality filtering.

Heuristic extraction (see :mod:`knowledge_graph_pkg.extraction`) trades
precision for recall: it fires on every sentence, so alongside genuine
facts it emits noise where a sentence-initial word ("The", "This",
"However", "Being") is mistaken for a subject, or where the object is a
run-on clause rather than a crisp answer.

:class:`FactQualityFilter` scores a fact on a few cheap, dependency-free
signals and rejects the obvious junk so the distilled output is clean
enough to use as training data.
"""

import re
from typing import Any, Dict, List

# Words that are never valid stand-alone subjects: determiners, pronouns,
# conjunctions, prepositions, and discourse/adverbial sentence openers.
_SUBJECT_STOPWORDS = {
    # determiners / pronouns
    "the", "a", "an", "this", "that", "these", "those", "there", "here",
    "it", "they", "we", "you", "he", "she", "one", "some", "any", "each",
    "every", "all", "both", "either", "neither", "another",
    # conjunctions / connectives
    "and", "or", "but", "however", "therefore", "thus", "moreover",
    "furthermore", "nevertheless", "meanwhile", "instead", "also",
    # subordinators / adverbial openers
    "if", "when", "while", "after", "before", "because", "although",
    "though", "since", "unless", "until", "whereas", "as", "so", "then",
    "once", "whenever", "wherever",
    # prepositions
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
    "over", "under", "into", "onto", "without", "within", "through",
    "during", "between", "among", "across", "toward",
    # bare verbs / chapter scaffolding that show up as openers
    "chapter", "section", "figure", "table",
}


class FactQualityFilter:
    """Score and filter extracted facts for downstream distillation.

    Args:
        max_object_len: Reject facts whose object string is longer than
            this (characters). Crisp answers are short noun phrases; long
            objects are usually run-on clauses. Default 80.
        min_subject_len: Reject facts whose subject is shorter than this
            (characters), which filters stray single letters. Default 2.
    """

    def __init__(self, max_object_len: int = 80, min_subject_len: int = 2,
                 require_entity_subject: bool = False):
        self.max_object_len = max_object_len
        self.min_subject_len = min_subject_len
        self.require_entity_subject = require_entity_subject

    # ------------------------------------------------------------------ #
    def is_acceptable(self, fact: Dict[str, Any]) -> bool:
        """Return True if the fact passes all quality gates."""
        subject = (fact.get("subject") or "").strip()
        object_ = (fact.get("object") or "").strip()

        # Non-empty subject and object.
        if not subject or not object_:
            return False
        if len(subject) < self.min_subject_len:
            return False

        # Subject head word must not be a stopword / sentence opener.
        head = subject.split()[0].lower().strip(".,'\"")
        if head in _SUBJECT_STOPWORDS:
            return False

        # Subject should look like a proper noun (capitalized) -- the
        # extractor only produces meaningful subjects when they are named
        # entities. Gerunds ("Developing") are capitalized too, so also
        # reject single-word -ing openers that aren't part of a proper name.
        if not subject[:1].isupper():
            return False
        words = subject.split()
        if len(words) == 1 and re.match(r"^[A-Z][a-z]+ing$", words[0]):
            return False

        # Strict mode: require the subject to be a named entity. The
        # extractor tags genuine named entities as ENTITY; abstract single
        # nouns (CONCEPT) and bare copula fragments are dropped. A multi-word
        # capitalized proper name is also accepted as an entity even if the
        # type wasn't set.
        if self.require_entity_subject:
            stype = fact.get("subject_type")
            multiword_proper = len(words) >= 2 and all(w[:1].isupper() for w in words)
            if stype != "ENTITY" and not multiword_proper:
                return False

        # Object must be reasonably short (a crisp answer, not a clause).
        if len(object_) > self.max_object_len:
            return False

        return True

    def score(self, fact: Dict[str, Any]) -> float:
        """Return a 0..1 quality score (higher is better).

        Cheap heuristic blend: acceptance, subject looking like an entity,
        and object brevity. Used for ranking and for monotonicity in tests.
        """
        if not self.is_acceptable(fact):
            # Still return a small gradient so clearly-worse facts rank lower.
            subject = (fact.get("subject") or "").strip()
            object_ = (fact.get("object") or "").strip()
            penalty = 0.0
            if subject and subject.split()[0].lower() not in _SUBJECT_STOPWORDS:
                penalty += 0.1
            if object_ and len(object_) <= self.max_object_len:
                penalty += 0.1
            return penalty

        object_ = fact["object"].strip()
        # Brevity bonus: shorter objects score higher (capped).
        brevity = max(0.0, 1.0 - len(object_) / max(1, self.max_object_len))
        entity_bonus = 0.3 if fact.get("subject_type") == "ENTITY" else 0.0
        return min(1.0, 0.5 + 0.2 * brevity + entity_bonus)

    def filter(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return only the acceptable facts, preserving order."""
        return [f for f in facts if self.is_acceptable(f)]
