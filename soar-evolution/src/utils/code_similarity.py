"""Code similarity metrics using token-level comparison."""

from __future__ import annotations

import re
from typing import List, Set


def normalize_code(code: str) -> str:
    """Normalize code for comparison (remove comments, whitespace)."""
    lines = code.splitlines()
    normalized = []
    for line in lines:
        # Remove inline comments
        line = re.sub(r"#.*$", "", line)
        # Strip whitespace
        stripped = line.strip()
        if stripped:
            normalized.append(stripped)
    return "\n".join(normalized)


def tokenize(code: str) -> List[str]:
    """Simple tokenization of Python code."""
    # Split on whitespace and punctuation
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\s]", code)
    return tokens


def code_similarity(code_a: str, code_b: str) -> float:
    """Compute similarity between two code strings (0.0-1.0).

    Uses Jaccard similarity on token n-grams.
    """
    if code_a == code_b:
        return 1.0

    if not code_a or not code_b:
        return 0.0

    norm_a = normalize_code(code_a)
    norm_b = normalize_code(code_b)

    if norm_a == norm_b:
        return 1.0

    # Token-level bigram Jaccard similarity
    tokens_a = tokenize(norm_a)
    tokens_b = tokenize(norm_b)

    if not tokens_a or not tokens_b:
        return 0.0

    ngrams_a = _make_ngrams(tokens_a, 2)
    ngrams_b = _make_ngrams(tokens_b, 2)

    if not ngrams_a and not ngrams_b:
        # Fall back to unigram comparison for very short code
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)

    return intersection / union if union > 0 else 0.0


def _make_ngrams(tokens: List[str], n: int) -> Set[tuple]:
    """Create n-gram set from token list."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def structural_similarity(code_a: str, code_b: str) -> float:
    """Compare structural similarity (ignoring variable names)."""
    # Replace identifiers with placeholders
    def anonymize(code: str) -> str:
        # Keep keywords and builtins, replace other identifiers
        keywords = {
            "def", "return", "if", "else", "elif", "for", "while",
            "in", "range", "len", "True", "False", "None", "and",
            "or", "not", "import", "from", "class", "try", "except",
            "finally", "with", "as", "yield", "lambda", "pass",
            "break", "continue", "raise", "append", "extend",
        }
        tokens = tokenize(code)
        result = []
        seen = {}
        counter = 0
        for token in tokens:
            if re.match(r"^[a-zA-Z_]", token) and token not in keywords:
                if token not in seen:
                    seen[token] = f"VAR{counter}"
                    counter += 1
                result.append(seen[token])
            else:
                result.append(token)
        return " ".join(result)

    anon_a = anonymize(normalize_code(code_a))
    anon_b = anonymize(normalize_code(code_b))

    tokens_a = anon_a.split()
    tokens_b = anon_b.split()

    ngrams_a = _make_ngrams(tokens_a, 2)
    ngrams_b = _make_ngrams(tokens_b, 2)

    if not ngrams_a and not ngrams_b:
        return 1.0 if tokens_a == tokens_b else 0.0

    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)

    return intersection / union if union > 0 else 0.0
