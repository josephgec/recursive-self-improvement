"""Stylometric feature extraction for AI-generated text detection.

Computes 12 statistical features that capture writing-style differences
between human-authored and machine-generated text, using only regex-based
tokenization (no external NLP libraries required).
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Top 100 English function words (determiners, prepositions, pronouns,
# auxiliary verbs, conjunctions, etc.).
FUNCTION_WORDS: frozenset[str] = frozenset(
    [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "some", "could", "them", "see", "other", "than", "then", "now",
        "look", "only", "come", "its", "over", "think", "also", "back",
        "after", "use", "two", "how", "our", "work", "first", "well",
        "way", "even", "new", "want", "because", "any", "these", "give",
        "most", "us", "is", "are", "was", "were", "been", "being", "had",
        "has", "did", "does",
    ]
)

# Conjunctions used for conjunction_rate.
CONJUNCTIONS: frozenset[str] = frozenset(
    [
        "and", "but", "or", "nor", "for", "yet", "so",
        "although", "because", "since", "unless", "while",
        "whereas", "however", "therefore", "moreover",
        "furthermore", "nevertheless", "consequently",
        "meanwhile", "otherwise", "instead", "besides",
    ]
)

# Regex for passive voice heuristic:
# Matches "was/were/been/being" followed by a word that looks like a
# past participle (ends in -ed, -en, -t, -wn, -ng, etc.).
_PASSIVE_RE = re.compile(
    r"\b(?:was|were|been|being|is|are|am)\s+"
    r"(?:\w+\s+){0,3}"              # allow up to 3 intervening adverbs
    r"(\w+ed|(\w+en)|built|sent|kept|made|done|gone|shown|known|grown|"
    r"drawn|thrown|written|driven|given|taken|broken|chosen|spoken|stolen|"
    r"frozen|forgotten|gotten|hidden|ridden|risen|shaken|awoken)\b",
    re.IGNORECASE,
)

# Simple sentence splitter: split on ., !, or ? followed by whitespace
# or end of string.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Word tokenizer: sequences of alphanumeric + apostrophe characters.
_WORD_RE = re.compile(r"[a-zA-Z'\u2019]+")

# Punctuation characters.
_PUNCT_RE = re.compile(r"[.,;:!?\"'()\[\]{}\-\u2013\u2014/]")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_stylometric_features(text: str) -> dict[str, float]:
    """Extract 12 stylometric features from *text*.

    Returns a dict with the following keys:

    - ``vocabulary_richness``: type-token ratio
    - ``hapax_ratio``: words appearing exactly once / total words
    - ``sentence_length_std``: standard deviation of sentence lengths
    - ``sentence_length_mean``: mean sentence length (in words)
    - ``paragraph_length_std``: standard deviation of paragraph lengths
    - ``yules_k``: Yule's K measure of vocabulary richness
    - ``function_word_ratio``: fraction of words that are function words
    - ``punctuation_ratio``: punctuation marks per sentence
    - ``avg_word_length``: mean word length in characters
    - ``conjunction_rate``: conjunctions per sentence
    - ``passive_voice_ratio``: estimated passive constructions / total sentences
    - ``repetition_score``: fraction of 4-grams appearing more than once
    """
    # ------------------------------------------------------------------
    # Tokenise
    # ------------------------------------------------------------------
    words = _WORD_RE.findall(text)
    words_lower = [w.lower() for w in words]
    n_words = len(words_lower)

    sentences = _split_sentences(text)
    n_sentences = max(len(sentences), 1)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # ------------------------------------------------------------------
    # 1. vocabulary_richness (type-token ratio)
    # ------------------------------------------------------------------
    n_types = len(set(words_lower))
    vocabulary_richness = n_types / n_words if n_words > 0 else 0.0

    # ------------------------------------------------------------------
    # 2. hapax_ratio
    # ------------------------------------------------------------------
    word_counts = Counter(words_lower)
    hapax = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio = hapax / n_words if n_words > 0 else 0.0

    # ------------------------------------------------------------------
    # 3-4. sentence_length_mean, sentence_length_std
    # ------------------------------------------------------------------
    sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    sentence_length_mean = _mean(sent_lengths)
    sentence_length_std = _std(sent_lengths)

    # ------------------------------------------------------------------
    # 5. paragraph_length_std
    # ------------------------------------------------------------------
    para_lengths = [len(_WORD_RE.findall(p)) for p in paragraphs]
    paragraph_length_std = _std(para_lengths)

    # ------------------------------------------------------------------
    # 6. Yule's K
    # ------------------------------------------------------------------
    yules_k = _compute_yules_k(word_counts, n_words)

    # ------------------------------------------------------------------
    # 7. function_word_ratio
    # ------------------------------------------------------------------
    n_function = sum(1 for w in words_lower if w in FUNCTION_WORDS)
    function_word_ratio = n_function / n_words if n_words > 0 else 0.0

    # ------------------------------------------------------------------
    # 8. punctuation_ratio (per sentence)
    # ------------------------------------------------------------------
    n_punct = len(_PUNCT_RE.findall(text))
    punctuation_ratio = n_punct / n_sentences

    # ------------------------------------------------------------------
    # 9. avg_word_length
    # ------------------------------------------------------------------
    avg_word_length = (
        sum(len(w) for w in words) / n_words if n_words > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # 10. conjunction_rate (per sentence)
    # ------------------------------------------------------------------
    n_conj = sum(1 for w in words_lower if w in CONJUNCTIONS)
    conjunction_rate = n_conj / n_sentences

    # ------------------------------------------------------------------
    # 11. passive_voice_ratio
    # ------------------------------------------------------------------
    n_passive = sum(1 for _ in _PASSIVE_RE.finditer(text))
    passive_voice_ratio = n_passive / n_sentences

    # ------------------------------------------------------------------
    # 12. repetition_score (fraction of 4-grams appearing > 1 time)
    # ------------------------------------------------------------------
    repetition_score = _ngram_repetition(words_lower, n=4)

    return {
        "vocabulary_richness": vocabulary_richness,
        "hapax_ratio": hapax_ratio,
        "sentence_length_std": sentence_length_std,
        "sentence_length_mean": sentence_length_mean,
        "paragraph_length_std": paragraph_length_std,
        "yules_k": yules_k,
        "function_word_ratio": function_word_ratio,
        "punctuation_ratio": punctuation_ratio,
        "avg_word_length": avg_word_length,
        "conjunction_rate": conjunction_rate,
        "passive_voice_ratio": passive_voice_ratio,
        "repetition_score": repetition_score,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using simple regex heuristics."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _compute_yules_k(word_counts: Counter, n_words: int) -> float:
    """Compute Yule's characteristic K.

    K = 10^4 * (M2 - N) / N^2
    where M2 = sum(i^2 * V_i) over all frequency classes i,
    and V_i is the number of word types occurring exactly i times.
    """
    if n_words == 0:
        return 0.0

    freq_spectrum = Counter(word_counts.values())  # freq -> n_types_with_that_freq
    m2 = sum(i * i * v_i for i, v_i in freq_spectrum.items())

    n = n_words
    denom = n * n
    if denom == 0:
        return 0.0

    k = 10_000.0 * (m2 - n) / denom
    return k


def _ngram_repetition(tokens: list[str], n: int = 4) -> float:
    """Fraction of *n*-grams that appear more than once."""
    if len(tokens) < n:
        return 0.0

    ngrams: list[tuple[str, ...]] = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i : i + n]))

    counts = Counter(ngrams)
    n_total = len(ngrams)
    n_repeated = sum(1 for ng, c in counts.items() if c > 1)
    total_unique = len(counts)

    return n_repeated / total_unique if total_unique > 0 else 0.0
