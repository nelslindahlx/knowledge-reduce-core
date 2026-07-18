"""
Question/answer generation from relation triples.

Turns a structured relation (subject, predicate, object) into a natural
question and answer so distilled facts become supervised training pairs.
For example the triple ("Marie Curie", "born_in", "Warsaw") becomes:

    Q: "Where was Marie Curie born?"
    A: "Warsaw"

This is intentionally template-based and dependency-free so it runs on any
machine without an NLP model.
"""

from typing import Any, Dict, Tuple


class QAGenerator:
    """Generate (question, answer) pairs from relation triples."""

    # Each template is a question phrased about the subject, whose answer is
    # the object. ``{s}`` is substituted with the subject.
    _QUESTION_TEMPLATES = {
        'born_in': "Where was {s} born?",
        'died_in': "Where did {s} die?",
        'located_in': "Where is {s} located?",
        'works_for': "Who or what does {s} work for?",
        'founded': "What did {s} found?",
        'owns': "What does {s} own?",
        'part_of': "What is {s} part of?",
        'married_to': "Who is {s} married to?",
        'parent_of': "Who is {s} a parent of?",
        'child_of': "Who is {s} a child of?",
        'discovered': "What did {s} discover?",
        'invented': "What did {s} invent?",
        'wrote': "What did {s} write?",
        'directed': "What did {s} direct?",
        'acted_in': "What did {s} act in?",
        'president_of': "What was {s} president of?",
        'director_of': "What was {s} director of?",
        'founder_of': "What was {s} founder of?",
        'author_of': "What was {s} author of?",
        'published': "When or where was {s} published?",
        'developed': "What did {s} develop?",
        'graduated': "What did {s} graduate with?",
        'collaborated': "What did {s} collaborate on?",
        'provided': "What did {s} provide?",
        'told': "What did {s} tell?",
        'is_a': "What is {s}?",
    }

    def generate(self, relation: Dict[str, Any]) -> Tuple[str, str]:
        """Return a ``(question, answer)`` pair for a relation triple.

        Args:
            relation: dict with ``subject``, ``predicate``, ``object`` keys.

        Returns:
            ``(question, answer)`` where the answer is the relation object.
        """
        subject = str(relation.get('subject', '')).strip()
        object_ = str(relation.get('object', '')).strip()
        predicate = relation.get('predicate', '')

        template = self._QUESTION_TEMPLATES.get(predicate)
        if template is None:
            # Generic but grammatical fallback; never leak raw underscores.
            readable = predicate.replace('_', ' ').strip()
            if readable:
                question = f"What does {subject} {readable}?"
            else:
                question = f"What is related to {subject}?"
        else:
            question = template.format(s=subject)

        return question, object_
