"""Utilities for generating synthetic contexts for testing and benchmarks."""

from __future__ import annotations

import random
import string


def generate_haystack(
    needle: str,
    haystack_size: int = 10000,
    position: float = 0.5,
    seed: int = 42,
) -> str:
    """Generate a haystack document with a needle hidden at a given relative position.

    Args:
        needle: The text to hide in the haystack.
        haystack_size: Approximate total character count.
        position: Where to insert the needle (0.0 = start, 1.0 = end).
        seed: Random seed for reproducibility.

    Returns:
        The full text with the needle inserted.
    """
    rng = random.Random(seed)
    filler_words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "and", "cat", "sat", "on", "mat", "while", "bird", "flew",
        "across", "sky", "into", "forest", "near", "river", "bank",
        "under", "old", "oak", "tree", "with", "green", "leaves",
    ]
    lines: list[str] = []
    total = 0
    while total < haystack_size:
        line_len = rng.randint(5, 15)
        line = " ".join(rng.choice(filler_words) for _ in range(line_len))
        lines.append(line)
        total += len(line) + 1  # +1 for newline

    insert_idx = max(0, min(len(lines) - 1, int(len(lines) * position)))
    lines.insert(insert_idx, needle)
    return "\n".join(lines)


def generate_document_collection(
    num_docs: int = 10,
    doc_size: int = 500,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate a collection of simple documents with titles and bodies."""
    rng = random.Random(seed)
    topics = [
        "Machine Learning", "Database Systems", "Network Security",
        "Cloud Computing", "Distributed Systems", "Compiler Design",
        "Operating Systems", "Computer Graphics", "Natural Language Processing",
        "Robotics", "Quantum Computing", "Blockchain", "IoT",
        "Data Warehousing", "Software Engineering", "Algorithms",
    ]
    filler = (
        "This section discusses various aspects of the topic including "
        "implementation details, performance considerations, and practical "
        "applications in modern systems. "
    )
    docs: list[dict[str, str]] = []
    for i in range(num_docs):
        topic = rng.choice(topics)
        title = f"Document {i + 1}: {topic}"
        body_parts = [f"Title: {title}\n"]
        remaining = doc_size - len(body_parts[0])
        while remaining > 0:
            body_parts.append(filler)
            remaining -= len(filler)
        docs.append({"title": title, "body": "".join(body_parts)[:doc_size]})
    return docs
