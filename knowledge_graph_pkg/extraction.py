"""
General-purpose, dependency-free relation extraction.

The original ``SemanticKnowledgeGraph`` extractor matched only a fixed set
of ~15 verb patterns and required a recognized named entity on both sides
of the verb. On dense real-world prose that produced almost nothing.

``SVOExtractor`` is a stronger pure-Python extractor. For each sentence it
finds a subject noun phrase, the main verb, and an object span, and emits
one or more relation triples. It handles:

* **Active SVO** -- "Robert Putnam wrote Bowling Alone."
* **Copula + role** -- "Carlsen was President of JCCC." -> president_of
* **Passive voice** -- "The book was published in 2006." -> published / 2006
* **Predicate nominal** -- "Civic engagement is a value choice." -> is_a
* **Multi-subject** -- "Alice and Bob graduated." -> two facts

It is heuristic, not a parser, but it extracts far more from ordinary
sentences than rigid verb-pattern matching, with no external dependencies.
"""

import re
from typing import Any, Dict, List, Optional

# Tokens that commonly start an object/complement and that we strip when
# they lead the object span.
_LEADING_STOPWORDS = {"the", "a", "an"}

# Auxiliary/copula forms.
_COPULA = {"is", "are", "was", "were", "be", "been", "being"}

# Verbs that, in passive constructions, carry a following preposition we want
# to fold into the predicate's object (e.g. "was published in 2006").
_PASSIVE_PREPS = {"in", "on", "at", "by", "from", "to", "with", "of"}

# Role nouns for the "X was <role> of Y" copula pattern.
_ROLE_NOUNS = {
    "president", "director", "founder", "author", "ceo", "chair",
    "chairman", "head", "leader", "professor", "member", "governor",
    "mayor", "secretary", "manager", "owner", "creator", "editor",
}

# Words that should never be treated as a subject on their own (or sentence-initial
# discourse/interrogative/auxiliary/quantifier openers).
_NON_SUBJECT = {
    "this", "that", "these", "those", "there", "here",
    "however", "therefore", "moreover", "furthermore", "meanwhile", "nevertheless",
    "instead", "also", "what", "why", "how", "who", "whom", "whose", "which",
    "where", "when", "while", "if", "although", "though", "because", "since",
    "unless", "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must",
    "many", "most", "no", "not", "few", "several", "such", "only", "even",
    "much", "more", "less", "none", "nothing", "nobody",
    "chapter", "section", "figure", "table",
}

_HEADING_CONNECTORS = {
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "and", "or", "the", "a", "an",
}

_NON_VERBS = {
    "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "as", "that", "which", "who", "the", "a", "an", "this",
    "these", "those", "my", "your", "his", "her", "its", "our", "their",
}


def _is_heading_line(sentence: str) -> bool:
    """Return True if *sentence* looks like a section header rather than prose."""
    s = sentence.strip()
    if not s or s.endswith(":"):
        return True
    if s.endswith((".", "!", "?")):
        return False
    words = [w.strip(",;:()\"'“”") for w in s.split() if w.strip(",;:()\"'“”")]
    if not words or len(words) > 12:
        return False
    sig = [w for w in words if w.lower() not in _HEADING_CONNECTORS]
    if not sig:
        return True
    cap = sum(1 for w in sig if w[:1].isupper())
    return (cap / len(sig)) >= 0.7

# Relative pronouns and subordinators that introduce a nested clause rather than
# continuing a multi-word noun phrase subject.
_CLAUSE_BREAKERS = {
    "which", "that", "who", "whom", "whose", "where", "when",
    "because", "although", "while", "since", "if",
}

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_WORD_RE = re.compile(r"[A-Za-z0-9'&-]+|[,]")

# Abbreviations whose trailing period must NOT be treated as a sentence end.
_ABBREVIATIONS = [
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.",
    "vs.", "etc.", "Inc.", "Corp.", "Ltd.", "Co.", "U.S.", "U.K.",
    "Ph.D.", "M.D.", "B.A.", "M.A.",
]
# Sentinel used to mask abbreviation periods during sentence splitting.
_DOT_SENTINEL = "\u0001"


def _normalize_predicate(verb: str) -> str:
    """Lowercase a verb and make it a single underscore-joined token."""
    return re.sub(r"\s+", "_", verb.strip().lower())


class SVOExtractor:
    """Extract subject-verb-object relation triples from text."""

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Return a list of relation dicts for all sentences in ``text``.

        Each dict has ``subject``, ``predicate``, ``object``,
        ``subject_type`` and ``object_type`` keys (types are coarse:
        ``ENTITY`` for capitalized spans, ``CONCEPT`` otherwise).
        """
        relations: List[Dict[str, Any]] = []
        for sentence in self._split_sentences(text):
            relations.extend(self._extract_sentence(sentence))
        return relations

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # Mask abbreviation periods and single-letter initials (e.g. "H. George")
        # so they don't trigger a sentence break.
        masked = text
        for abbr in _ABBREVIATIONS:
            masked = masked.replace(abbr, abbr.replace(".", _DOT_SENTINEL))
        masked = re.sub(r"\b([A-Z])\.(?=\s)", rf"\1{_DOT_SENTINEL}", masked)
        out = []
        for m in _SENTENCE_RE.finditer(masked):
            s = m.group(0).replace(_DOT_SENTINEL, ".").strip()
            if s:
                out.append(s)
        return out

    def _extract_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        stripped = sentence.strip()
        if not stripped or stripped.endswith("?") or _is_heading_line(stripped):
            return []

        # Strip a trailing period for cleaner objects.
        clean = stripped.rstrip(".!?")
        tokens = clean.split()
        if len(tokens) < 2:
            return []

        head0 = tokens[0].rstrip(",;:").lower()
        if head0 in _NON_SUBJECT:
            return []

        if tokens[0].lower() in {"the", "a", "an"} and len(tokens) >= 3:
            rest_after_art = tokens[1:]
            if rest_after_art[0][:1].isupper():
                subj_text, rest = self._take_subject(rest_after_art)
                if subj_text is None:
                    subj_text = rest_after_art[0].rstrip(",")
                    rest = rest_after_art[1:]
            else:
                subj_text = rest_after_art[0].rstrip(",")
                rest = rest_after_art[1:]
        else:
            subj_text, rest = self._take_subject(tokens)

        if subj_text is None or not rest:
            return []

        # If the captured subject is a single word (not a pronoun) followed by up
        # to a few non-clause words before a copula, fold them into the subject.
        # This handles "Civic engagement is a value choice." without swallowing
        # active verbs or relative clauses ("Albert Einstein discovered ...").
        if (
            rest
            and len(subj_text.split()) == 1
            and subj_text.lower() not in {"he", "she", "they", "it"}
        ):
            cop_idx = next((i for i, t in enumerate(rest) if t.lower() in _COPULA), None)
            if cop_idx is not None and 0 < cop_idx <= 3:
                extra = rest[:cop_idx]
                if all(
                    e.lower() not in _COPULA
                    and e.lower() not in _CLAUSE_BREAKERS
                    and not e.endswith(",")
                    for e in extra
                ):
                    subj_text = (subj_text + " " + " ".join(extra)).strip()
                    rest = rest[cop_idx:]

        if subj_text.split()[0].rstrip(",").lower() in _NON_SUBJECT:
            return []

        subjects = self._split_conjoined_subjects(subj_text)

        verb_idx = self._find_verb(rest)
        if verb_idx is None:
            return []

        verb = rest[verb_idx].lower()
        after = rest[verb_idx + 1:]

        triples = self._build_from_verb(verb, after)
        if triples is None:
            return []

        predicate, object_text = triples
        if not object_text:
            return []

        relations = []
        for subj in subjects:
            relations.append({
                "subject": subj,
                "subject_type": self._coarse_type(subj),
                "predicate": predicate,
                "object": object_text,
                "object_type": self._coarse_type(object_text),
                "text": stripped,
            })
        return relations

    # ------------------------------------------------------------------ #
    def _take_subject(self, tokens: List[str]):
        """Consume a leading subject noun phrase; return (subject, rest_tokens).

        The subject is a run of capitalized words (optionally joined by
        ``and``/``&`` and titles like ``Dr.``), or a single leading pronoun.
        """
        # Pronoun subject.
        if tokens[0].lower() in {"he", "she", "they", "it"} and tokens[0][0].isupper():
            return tokens[0], tokens[1:]

        i = 0
        n = len(tokens)
        captured = []
        while i < n:
            w = tokens[i]
            bare = w.rstrip(".,")
            is_cap = bare[:1].isupper() and bare[:1].isalpha()
            is_join = w.lower() in {"and", "&"} and captured  # allow conjunction inside subject
            if is_cap:
                captured.append(w)
                i += 1
                continue
            if is_join:
                # Only treat as part of subject if the next token is also capitalized.
                if i + 1 < n and tokens[i + 1][:1].isupper():
                    captured.append(w)
                    i += 1
                    continue
                break
            break

        if not captured:
            return None, tokens

        subject = " ".join(captured).rstrip(",")
        # Don't let a copula/verb get swallowed; subject is purely the cap run.
        return subject, tokens[i:]

    @staticmethod
    def _split_conjoined_subjects(subject: str) -> List[str]:
        parts: List[str] = []
        for chunk in re.split(r"\s+and\s+", subject):
            chunk = chunk.strip()
            if not chunk:
                continue
            # Keep single-word company/duo names joined by '&' intact (e.g.
            # "Johnson & Johnson"), only splitting '&' when both sides are
            # distinct multi-word entities.
            amp_parts = [p.strip() for p in re.split(r"\s+&\s+", chunk) if p.strip()]
            if (
                len(amp_parts) == 2
                and len(amp_parts[0].split()) >= 2
                and len(amp_parts[1].split()) >= 2
                and amp_parts[0].lower() != amp_parts[1].lower()
            ):
                parts.extend(amp_parts)
            else:
                parts.append(chunk)
        return parts

    def _find_verb(self, tokens: List[str]) -> Optional[int]:
        """Find the index of the main verb in the post-subject tokens."""
        for i, tok in enumerate(tokens):
            low = tok.lower()
            if low in _COPULA:
                return i
            # A lowercase alphabetic word that isn't a preposition/determiner/conjunction
            # is treated as the main verb.
            if low.isalpha() and low not in _NON_VERBS:
                return i
        return None

    def _build_from_verb(self, verb: str, after: List[str]):
        """Return (predicate, object_text) for the verb and trailing tokens."""
        if not after:
            return None

        # Copula handling: is/are/was/were ...
        if verb in _COPULA:
            return self._build_copula(after)

        # Active verb: predicate is the verb; object is the trailing noun
        # phrase, with a leading preposition folded away.
        obj_tokens = after
        if obj_tokens and obj_tokens[0].lower() in {"with", "to", "for", "in", "on", "at", "by", "from"}:
            obj_tokens = obj_tokens[1:]
        object_text = self._clean_object(obj_tokens)
        return _normalize_predicate(verb), object_text

    def _build_copula(self, after: List[str]):
        """Handle 'was President of Y', 'was published in Y', 'is a Y'."""
        first = after[0].lower()

        # Role: "<role> of Y" -> predicate role_of, object Y
        if first in _ROLE_NOUNS and len(after) >= 3 and after[1].lower() == "of":
            obj = self._clean_object(after[2:])
            return f"{first}_of", obj

        # Passive: "<participle> <prep> Y" -> predicate, object Y.
        # Covers regular "-ed" participles and common irregulars (born, etc.).
        _IRREGULAR_PARTICIPLES = {
            "born", "written", "made", "built", "found", "given", "taken",
            "known", "held", "led", "begun", "set", "put", "sent",
        }
        if len(after) >= 2 and after[1].lower() in _PASSIVE_PREPS and (
            first.endswith("ed") or first in _IRREGULAR_PARTICIPLES
        ):
            obj = self._clean_object(after[2:])
            predicate = first
            # Fold the preposition into the predicate for natural phrasing
            # of common cases (born_in, based_in, located_in).
            if first in {"born", "based", "located", "situated"}:
                predicate = f"{first}_{after[1].lower()}"
            return _normalize_predicate(predicate), obj

        # Predicate nominal: "a/an/the Y" or bare noun -> is_a
        if first in {"a", "an"}:
            obj = self._clean_object(after[1:])
            return "is_a", obj
        if first == "the":
            obj = self._clean_object(after[1:])
            return "is", obj

        # Generic copula complement.
        obj = self._clean_object(after)
        return "is", obj

    @staticmethod
    def _clean_object(tokens: List[str]) -> str:
        toks = list(tokens)
        while toks and toks[0].lower() in _LEADING_STOPWORDS:
            toks = toks[1:]
        text = " ".join(toks).strip().rstrip(".,;:!?")
        return text

    @staticmethod
    def _coarse_type(text: str) -> str:
        head = text.split()[0] if text else ""
        if head[:1].isupper():
            return "ENTITY"
        return "CONCEPT"
