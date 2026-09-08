"""Extractive summarization that adds no inference cost.

Key sentences are chosen from the transcript itself, scored by how much rare
vocabulary each one carries. Nothing here calls a model, so a summary costs
microseconds and the text never leaves the process.

Scoring deliberately avoids per-language stopword lists: the most frequent terms
in any transcript are that language's function words, so discarding the common
head of the frequency distribution approximates stopword removal in all supported
languages. Scripts that do not separate words with spaces fall back to character
bigrams, which behave the same way statistically.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

#: Below this a transcript is its own summary and key points would just repeat it.
_MIN_SENTENCES_FOR_KEY_POINTS = 4

#: Keep a summary glanceable rather than a second transcript.
_MAX_KEY_POINTS = 5
_SENTENCES_PER_KEY_POINT = 3

#: Share of the most frequent terms treated as function words.
_COMMON_TERM_RATIO = 0.25

# Full-width marks carry no trailing space in CJK text, so they end a sentence on
# their own. ASCII and Arabic/Devanagari marks require following whitespace, which
# keeps decimals such as "3.5 kg" intact.
_FULL_WIDTH_STOPS = "。！？"  # noqa: RUF001
_OTHER_STOPS = ".!?؟।"
_SENTENCE_BOUNDARY = re.compile(rf"(?<=[{_FULL_WIDTH_STOPS}])\s*|(?<=[{_OTHER_STOPS}])\s+|\n+")
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_SPACELESS_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR")


def summarize(transcript: str) -> tuple[str, ...]:
    """Return the transcript's most distinctive sentences, in spoken order."""

    sentences = split_sentences(transcript)
    if len(sentences) < _MIN_SENTENCES_FOR_KEY_POINTS:
        return ()

    tokens_by_index = [_tokenize(sentence) for sentence in sentences]
    weights = _term_weights(tokens_by_index)
    wanted = min(_MAX_KEY_POINTS, max(1, len(sentences) // _SENTENCES_PER_KEY_POINT))

    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (-_score(tokens_by_index[index], weights), index),
    )
    return tuple(sentences[index] for index in sorted(ranked[:wanted]))


def split_sentences(transcript: str) -> tuple[str, ...]:
    """Split on sentence punctuation the recognizer already emits."""

    parts = (part.strip() for part in _SENTENCE_BOUNDARY.split(transcript))
    return tuple(part for part in parts if part)


def _term_weights(tokens_by_index: Sequence[Sequence[str]]) -> dict[str, float]:
    """Weight terms by frequency after discarding the common function-word head."""

    counts: Counter[str] = Counter()
    for tokens in tokens_by_index:
        counts.update(tokens)
    if not counts:
        return {}

    ordered = counts.most_common()
    dropped = int(len(ordered) * _COMMON_TERM_RATIO)
    return {term: float(count) for term, count in ordered[dropped:]}


def _score(tokens: Sequence[str], weights: dict[str, float]) -> float:
    """Favour sentences dense in distinctive terms, not merely long ones."""

    if not tokens:
        return 0.0
    total = sum(weights.get(token, 0.0) for token in set(tokens))
    return total / math.sqrt(len(tokens))


def _tokenize(text: str) -> tuple[str, ...]:
    words = _WORD_PATTERN.findall(text.casefold())
    if not words:
        return ()
    if not _is_spaceless(text):
        return tuple(words)
    # Words never separate in these scripts, so one "word" is a whole run of
    # characters; character bigrams recover comparable term statistics.
    joined = "".join(words)
    if len(joined) < 2:
        return (joined,)
    return tuple(joined[index : index + 2] for index in range(len(joined) - 1))


def _is_spaceless(text: str) -> bool:
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if any(script in name for script in _SPACELESS_SCRIPTS):
            return True
    return False
