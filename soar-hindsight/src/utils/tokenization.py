"""Simple token counting utilities (word-based approximation)."""

from __future__ import annotations

from typing import List


def count_tokens(text: str) -> int:
    """Approximate token count using word-based splitting.

    Uses a simple heuristic: ~0.75 words per token (GPT-like).
    This avoids requiring tiktoken as a dependency.
    """
    if not text:
        return 0
    # Split on whitespace and punctuation boundaries
    words = text.split()
    # Approximate: each word is ~1.3 tokens on average
    return max(1, int(len(words) * 1.3))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens tokens."""
    if not text:
        return text
    words = text.split()
    # Inverse of the counting heuristic
    max_words = int(max_tokens / 1.3)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def count_tokens_batch(texts: List[str]) -> List[int]:
    """Count tokens for a batch of texts."""
    return [count_tokens(t) for t in texts]
