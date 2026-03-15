"""OlympiadBench benchmark loader."""

from __future__ import annotations

from typing import Any

from src.benchmarks.math500 import BenchmarkProblem
from src.utils.logging import get_logger

logger = get_logger("benchmarks.olympiad")

# Synthetic fallback problems used when the real dataset is unavailable.
_SYNTHETIC_PROBLEMS: list[dict[str, Any]] = [
    {
        "problem": (
            "Find all positive integers n such that n^2 + 1 is divisible by n + 1."
        ),
        "answer": "1",
        "subject": "number_theory",
        "difficulty": 3,
    },
    {
        "problem": (
            "Let a, b, c be positive reals with a + b + c = 1. "
            "Prove that a^2 + b^2 + c^2 >= 1/3 and find the minimum value."
        ),
        "answer": "1/3",
        "subject": "algebra",
        "difficulty": 4,
    },
    {
        "problem": (
            "In triangle ABC, the incircle touches sides BC, CA, AB at D, E, F "
            "respectively. If BD = 3, CE = 4, and AF = 5, find the area of "
            "triangle ABC."
        ),
        "answer": "12*sqrt(6)",
        "subject": "geometry",
        "difficulty": 5,
    },
    {
        "problem": (
            "How many ways can you tile a 2 x 10 rectangle with 1 x 2 dominoes?"
        ),
        "answer": "89",
        "subject": "combinatorics",
        "difficulty": 3,
    },
    {
        "problem": (
            "Find the sum of the series: sum_{k=1}^{infinity} k / 2^k."
        ),
        "answer": "2",
        "subject": "calculus",
        "difficulty": 3,
    },
    {
        "problem": (
            "A box contains 5 red balls and 3 blue balls. Two balls are drawn "
            "without replacement. What is the probability that both are red?"
        ),
        "answer": "5/14",
        "subject": "probability",
        "difficulty": 2,
    },
    {
        "problem": (
            "Find the remainder when 7^{83} is divided by 15."
        ),
        "answer": "13",
        "subject": "number_theory",
        "difficulty": 3,
    },
    {
        "problem": (
            "Determine the number of 5-element subsets of {1,2,...,10} that "
            "contain no two consecutive integers."
        ),
        "answer": "6",
        "subject": "combinatorics",
        "difficulty": 4,
    },
    {
        "problem": (
            "If f(x) = x^3 - 3x + 1, find the number of real roots of f(x) = 0."
        ),
        "answer": "3",
        "subject": "algebra",
        "difficulty": 3,
    },
    {
        "problem": (
            "Let P(x) = x^4 - 4x^3 + 6x^2 - 4x + 1. Compute P(2)."
        ),
        "answer": "1",
        "subject": "algebra",
        "difficulty": 2,
    },
]


class OlympiadBenchLoader:
    """Load problems from OlympiadBench.

    Attempts to load from HuggingFace first, then falls back to a small
    set of synthetic olympiad-style problems for offline/testing use.
    """

    def __init__(
        self,
        dataset_name: str = "olympiad_bench",
        split: str = "test",
        num_problems: int | None = None,
        cache_dir: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.num_problems = num_problems
        self.cache_dir = cache_dir

    def load(self) -> list[BenchmarkProblem]:
        """Load olympiad problems.

        Tries HuggingFace ``datasets`` first; falls back to built-in
        synthetic problems if loading fails.
        """
        problems = self._try_load_huggingface()
        if problems:
            return problems
        logger.info("Falling back to synthetic olympiad problems")
        return self._load_synthetic()

    # ── HuggingFace loader ──────────────────────────────────────────

    def _try_load_huggingface(self) -> list[BenchmarkProblem]:
        """Try loading from HuggingFace. Return empty list on failure."""
        try:
            from datasets import load_dataset

            ds = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning("Could not load OlympiadBench from HuggingFace: %s", exc)
            return []

        problems: list[BenchmarkProblem] = []
        for i, row in enumerate(ds):
            if self.num_problems is not None and i >= self.num_problems:
                break

            answer = str(row.get("answer", row.get("expected_answer", "")))
            subject = str(row.get("subject", row.get("type", ""))).lower().strip()
            difficulty = int(row.get("difficulty", row.get("level", 0)))

            problems.append(
                BenchmarkProblem(
                    problem_id=f"olympiad_{i:04d}",
                    problem=str(row.get("problem", row.get("question", ""))),
                    expected_answer=answer,
                    subject=subject,
                    difficulty=difficulty,
                    solution=str(row.get("solution", "")),
                )
            )

        logger.info("Loaded %d OlympiadBench problems from HuggingFace", len(problems))
        return problems

    # ── synthetic fallback ──────────────────────────────────────────

    def _load_synthetic(self) -> list[BenchmarkProblem]:
        """Return built-in synthetic olympiad problems."""
        problems: list[BenchmarkProblem] = []
        limit = self.num_problems or len(_SYNTHETIC_PROBLEMS)
        for i, entry in enumerate(_SYNTHETIC_PROBLEMS[:limit]):
            problems.append(
                BenchmarkProblem(
                    problem_id=f"olympiad_synth_{i:04d}",
                    problem=entry["problem"],
                    expected_answer=entry["answer"],
                    subject=entry["subject"],
                    difficulty=entry["difficulty"],
                    solution="",
                )
            )
        logger.info("Loaded %d synthetic olympiad problems", len(problems))
        return problems

    def load_by_subject(self, subject: str) -> list[BenchmarkProblem]:
        """Load problems filtered by subject."""
        all_problems = self.load()
        subject_lower = subject.lower().strip()
        return [p for p in all_problems if p.subject == subject_lower]

    def load_by_difficulty(self, difficulty: int) -> list[BenchmarkProblem]:
        """Load problems filtered by difficulty level."""
        all_problems = self.load()
        return [p for p in all_problems if p.difficulty == difficulty]
