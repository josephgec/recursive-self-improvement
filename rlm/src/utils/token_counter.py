"""Token counting utilities using a simple word-based estimator."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 chars per token heuristic."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_tokens(text: str) -> int:
    """Count tokens by splitting on whitespace and punctuation boundaries.

    More accurate than estimate_tokens but still does not require a tokenizer.
    """
    if not text:
        return 0
    # Rough: split on whitespace, then each word is ~1.3 tokens on average
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.3))
