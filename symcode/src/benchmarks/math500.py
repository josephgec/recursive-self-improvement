"""MATH-500 benchmark loader from hendrycks/competition_math."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging import get_logger

logger = get_logger("benchmarks.math500")


@dataclass
class BenchmarkProblem:
    """A single benchmark problem."""

    problem_id: str
    problem: str
    expected_answer: str
    subject: str = ""
    difficulty: int = 0
    solution: str = ""


def _extract_boxed(text: str) -> str | None:
    r"""Extract content from \boxed{...}, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    idx += len(r"\boxed{")
    depth = 1
    end = idx
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    if depth != 0:
        return None
    return text[idx : end - 1].strip()


class MATH500Loader:
    """Load problems from the MATH-500 dataset (hendrycks/competition_math).

    Falls back gracefully if the ``datasets`` library or the HuggingFace
    dataset is not available.
    """

    DATASET_NAME = "hendrycks/competition_math"
    DEFAULT_SPLIT = "test"

    def __init__(
        self,
        dataset_name: str | None = None,
        split: str | None = None,
        num_problems: int | None = 500,
        cache_dir: str | None = None,
    ):
        self.dataset_name = dataset_name or self.DATASET_NAME
        self.split = split or self.DEFAULT_SPLIT
        self.num_problems = num_problems
        self.cache_dir = cache_dir

    # ── core loader ─────────────────────────────────────────────────

    def load(self) -> list[BenchmarkProblem]:
        """Load benchmark problems from HuggingFace datasets.

        Returns an empty list if the dataset cannot be loaded.
        """
        try:
            from datasets import load_dataset

            ds = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning("Could not load MATH dataset: %s", exc)
            return []

        problems: list[BenchmarkProblem] = []
        for i, row in enumerate(ds):
            if self.num_problems is not None and i >= self.num_problems:
                break

            solution_text = row.get("solution", "")
            answer = _extract_boxed(solution_text)
            if answer is None:
                # Try the explicit answer field if present
                answer = str(row.get("answer", ""))

            subject = row.get("type", row.get("subject", ""))

            # MATH dataset "level" field is like "Level 3" -- extract int
            level_str = str(row.get("level", row.get("difficulty", "0")))
            level_match = re.search(r"(\d+)", level_str)
            difficulty = int(level_match.group(1)) if level_match else 0

            problems.append(
                BenchmarkProblem(
                    problem_id=f"math500_{i:04d}",
                    problem=row.get("problem", ""),
                    expected_answer=answer or "",
                    subject=subject.lower().strip() if subject else "",
                    difficulty=difficulty,
                    solution=solution_text,
                )
            )

        logger.info("Loaded %d MATH-500 problems", len(problems))
        return problems

    # ── filtered loaders ────────────────────────────────────────────

    def load_by_subject(self, subject: str) -> list[BenchmarkProblem]:
        """Load problems filtered by subject (e.g. 'algebra')."""
        all_problems = self.load()
        subject_lower = subject.lower().strip()
        return [p for p in all_problems if p.subject == subject_lower]

    def load_by_difficulty(self, difficulty: int) -> list[BenchmarkProblem]:
        """Load problems filtered by difficulty level."""
        all_problems = self.load()
        return [p for p in all_problems if p.difficulty == difficulty]
