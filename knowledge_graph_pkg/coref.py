"""
Lightweight pronoun / coreference resolution.

Sentence-by-sentence extraction loses pronoun antecedents: given
"Marie Curie was born in Warsaw. She discovered radium." the second
sentence yields a fact about "She" instead of "Marie Curie".

This module provides a dependency-free heuristic resolver that rewrites a
leading subject pronoun (She/He/They/It) with the most recent compatible
named entity from earlier in the text. It is intentionally conservative:
when no suitable antecedent is known it leaves the pronoun untouched rather
than guessing.

This is NOT a full coreference system (no spaCy/neuralcoref). It targets
the common "pronoun at the start of a sentence" case that most degrades
the quality of auto-extracted training data.
"""

import re
from typing import List, Optional

# Subject pronouns we attempt to resolve, mapped to the antecedent kind.
_PRONOUN_KIND = {
    "he": "person",
    "she": "person",
    "they": "plural",
    "it": "thing",
}

# Person phrase: two or more capitalized words, not led by "The".
_PERSON_RE = re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b')

# Generic proper-noun phrase: optional leading "The" + capitalized word(s).
_ENTITY_RE = re.compile(r'\b((?:The\s)?[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b')


def _sentences_with_spans(text: str):
    """Yield ``(chunk, leading_ws_len)`` chunks that reconstruct ``text``.

    Each chunk includes any leading whitespace so that concatenating the
    chunks (after optional in-place edits) reproduces the original text
    exactly when no edit is made.
    """
    for match in re.finditer(r'\s*[^.!?]+[.!?]?', text):
        chunk = match.group(0)
        leading = len(chunk) - len(chunk.lstrip())
        yield chunk, leading


def _extract_people(text: str) -> List[str]:
    return [m.group(1) for m in _PERSON_RE.finditer(text)]


def _extract_entities(text: str) -> List[str]:
    return [m.group(1) for m in _ENTITY_RE.finditer(text)]


def resolve_coreferences(text: str) -> str:
    """Resolve leading subject pronouns to recent named-entity antecedents.

    Args:
        text: Input text, potentially spanning multiple sentences.

    Returns:
        Text with resolvable leading pronouns replaced by their antecedent.
        Unresolvable pronouns (no prior antecedent) are left unchanged.
    """
    last_person: Optional[str] = None
    last_entity: Optional[str] = None
    last_plural: Optional[str] = None  # entity starting with "The" or ending in 's'

    out_parts: List[str] = []

    for chunk, leading in _sentences_with_spans(text):
        ws = chunk[:leading]
        body = chunk[leading:]

        # Try to resolve a leading subject pronoun.
        m = re.match(r'(He|She|They|It)\b', body)
        if m:
            pronoun = m.group(1)
            kind = _PRONOUN_KIND[pronoun.lower()]
            antecedent = None
            if kind == "person":
                antecedent = last_person or last_entity
            elif kind == "plural":
                antecedent = last_plural or last_entity
            else:  # thing
                antecedent = last_entity

            if antecedent:
                body = antecedent + body[m.end():]

        out_parts.append(ws + body)

        # Update trackers from this (resolved) sentence for later sentences.
        for person in _extract_people(body):
            last_person = person
        for entity in _extract_entities(body):
            last_entity = entity
            if entity.startswith("The ") or entity.rstrip(".").endswith("s"):
                last_plural = entity

    return "".join(out_parts)
