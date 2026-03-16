"""Synthetic long-context task generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.context_generators import generate_haystack


@dataclass
class SyntheticTask:
    """A synthetic benchmark task."""
    task_id: str
    category: str
    query: str
    context: str
    expected_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticLongContextGenerator:
    """Generate synthetic long-context tasks for evaluation."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def needle_in_haystack(
        self,
        needle: str = "The secret password is: OPEN_SESAME_42",
        haystack_size: int = 10000,
        position: float = 0.5,
    ) -> SyntheticTask:
        """Generate a needle-in-haystack task."""
        context = generate_haystack(
            needle=needle,
            haystack_size=haystack_size,
            position=position,
            seed=self.seed,
        )
        return SyntheticTask(
            task_id=f"synthetic_needle_{int(position * 100)}",
            category="needle_in_haystack",
            query="Find the secret password hidden in the text.",
            context=context,
            expected_answer="OPEN_SESAME_42",
            metadata={"position": position, "haystack_size": haystack_size},
        )

    def multi_needle(
        self,
        needles: Optional[List[str]] = None,
        haystack_size: int = 15000,
    ) -> SyntheticTask:
        """Generate a multi-needle task."""
        if needles is None:
            needles = [
                "NEEDLE_A: value_alpha",
                "NEEDLE_B: value_beta",
                "NEEDLE_C: value_gamma",
            ]

        # Build haystack with multiple needles
        context = generate_haystack(
            needle=needles[0],
            haystack_size=haystack_size,
            position=0.2,
            seed=self.seed,
        )
        lines = context.split("\n")
        for i, needle in enumerate(needles[1:], 1):
            pos = int(len(lines) * (0.2 + 0.3 * i))
            pos = min(pos, len(lines))
            lines.insert(pos, needle)
        context = "\n".join(lines)

        expected = ", ".join(n.split(": ")[1] for n in needles)
        return SyntheticTask(
            task_id="synthetic_multi_needle",
            category="multi_needle",
            query="Find all NEEDLE values and list them.",
            context=context,
            expected_answer=expected,
            metadata={"num_needles": len(needles)},
        )

    def counting_task(
        self,
        target_word: str = "specific_marker",
        count: int = 7,
        haystack_size: int = 8000,
    ) -> SyntheticTask:
        """Generate a counting task: how many times does a word appear?"""
        import random
        rng = random.Random(self.seed)

        filler_words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "and", "sat", "on", "mat", "while", "flew", "across",
        ]
        lines: List[str] = []
        total_chars = 0
        markers_placed = 0

        while total_chars < haystack_size or markers_placed < count:
            line_len = rng.randint(5, 12)
            words = [rng.choice(filler_words) for _ in range(line_len)]
            if markers_placed < count and rng.random() < 0.15:
                insert_pos = rng.randint(0, len(words))
                words.insert(insert_pos, target_word)
                markers_placed += 1
            line = " ".join(words)
            lines.append(line)
            total_chars += len(line) + 1

        # Ensure we have exactly *count* occurrences
        text = "\n".join(lines)
        actual_count = text.split().count(target_word)
        # Adjust by appending or removing
        while actual_count < count:
            lines.append(f"Note: {target_word} appears here.")
            actual_count += 1
        text = "\n".join(lines)
        actual_count = text.split().count(target_word)

        return SyntheticTask(
            task_id="synthetic_counting",
            category="counting",
            query=f"How many times does '{target_word}' appear in the text?",
            context=text,
            expected_answer=str(actual_count),
            metadata={"target_word": target_word, "expected_count": actual_count},
        )

    def summarization_task(
        self,
        num_sections: int = 5,
    ) -> SyntheticTask:
        """Generate a summarization task with multiple sections."""
        sections = []
        topics = [
            ("Introduction", "This document introduces the concept of distributed systems."),
            ("Architecture", "The system uses a microservices architecture with event-driven communication."),
            ("Performance", "Benchmarks show 99th percentile latency under 50ms."),
            ("Security", "All data is encrypted at rest and in transit using AES-256."),
            ("Conclusion", "The system meets all requirements for production deployment."),
            ("Future Work", "Next steps include adding support for multi-region deployment."),
            ("References", "See RFC 7540 for HTTP/2 specification details."),
        ]
        used_topics = topics[:num_sections]
        for title, summary in used_topics:
            padding = generate_haystack(
                needle=summary,
                haystack_size=1000,
                position=0.3,
                seed=self.seed + hash(title) % 1000,
            )
            sections.append(f"## {title}\n{padding}")

        context = "\n\n".join(sections)
        expected = "; ".join(s for _, s in used_topics)

        return SyntheticTask(
            task_id="synthetic_summarization",
            category="summarization",
            query="Summarize the key point from each section.",
            context=context,
            expected_answer=expected,
            metadata={"num_sections": num_sections},
        )
