"""Benchmark runner: execute problems through SymCode and prose pipelines."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.benchmarks.math500 import BenchmarkProblem
from src.pipeline.code_generator import SymCodeGenerator
from src.pipeline.prose_baseline import ProseBaseline
from src.pipeline.router import TaskRouter
from src.verification.answer_checker import AnswerChecker
from src.verification.result_types import SolveResult
from src.verification.retry_loop import RetryLoop
from src.utils.logging import get_logger

logger = get_logger("benchmarks.runner")


@dataclass
class BenchmarkResult:
    """Aggregate results of a benchmark run."""

    symcode_results: list[SolveResult] = field(default_factory=list)
    prose_results: list[SolveResult] = field(default_factory=list)

    # Top-level accuracy
    symcode_accuracy: float = 0.0
    prose_accuracy: float = 0.0

    # Per-subject breakdown: subject -> {"symcode": float, "prose": float, "count": int}
    per_subject: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-difficulty breakdown: level -> {"symcode": float, "prose": float, "count": int}
    per_difficulty: dict[int, dict[str, Any]] = field(default_factory=dict)

    total_time: float = 0.0
    num_problems: int = 0

    def compute_summaries(self) -> None:
        """Compute accuracy and breakdown summaries from raw results."""
        # Top-level accuracy
        if self.symcode_results:
            correct = sum(1 for r in self.symcode_results if r.correct)
            self.symcode_accuracy = correct / len(self.symcode_results)

        if self.prose_results:
            correct = sum(1 for r in self.prose_results if r.correct)
            self.prose_accuracy = correct / len(self.prose_results)

        self.num_problems = max(len(self.symcode_results), len(self.prose_results))

        # Per-subject breakdown
        self._compute_breakdown("subject")
        # Per-difficulty breakdown
        self._compute_breakdown("difficulty")

    def _compute_breakdown(self, key: str) -> None:
        """Compute accuracy breakdown by subject or difficulty."""
        # Build groups from symcode results
        groups: dict[Any, dict[str, list[bool]]] = {}

        for r in self.symcode_results:
            val = getattr(r, "task_type", "") if key == "subject" else 0
            # We stash the subject/difficulty in task_type field for grouping
            if key == "subject":
                val = r.task_type or "unknown"
            else:
                val = r.num_attempts  # placeholder, replaced below

            if val not in groups:
                groups[val] = {"symcode": [], "prose": []}
            groups[val]["symcode"].append(r.correct)

        # For per-difficulty, we need to match by problem text
        # Build an index from problem -> difficulty
        # (We use the problem index ordering instead)
        if key == "difficulty":
            groups.clear()

        # Simpler approach: zip symcode and prose results since they share ordering
        for i, sr in enumerate(self.symcode_results):
            group_key: Any
            if key == "subject":
                group_key = sr.task_type or "unknown"
            else:
                # Difficulty is encoded in the SolveResult via the external caller
                group_key = getattr(sr, "_difficulty", 0)

            if group_key not in groups:
                groups[group_key] = {"symcode": [], "prose": []}
            groups[group_key]["symcode"].append(sr.correct)

            if i < len(self.prose_results):
                groups[group_key]["prose"].append(self.prose_results[i].correct)

        breakdown: dict[Any, dict[str, Any]] = {}
        for gk, data in groups.items():
            sym_vals = data["symcode"]
            prose_vals = data["prose"]
            breakdown[gk] = {
                "symcode": sum(sym_vals) / len(sym_vals) if sym_vals else 0.0,
                "prose": sum(prose_vals) / len(prose_vals) if prose_vals else 0.0,
                "count": len(sym_vals),
            }

        if key == "subject":
            self.per_subject = breakdown
        else:
            self.per_difficulty = {int(k) if isinstance(k, (int, float)) else k: v for k, v in breakdown.items()}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "symcode_accuracy": self.symcode_accuracy,
            "prose_accuracy": self.prose_accuracy,
            "num_problems": self.num_problems,
            "total_time": self.total_time,
            "per_subject": self.per_subject,
            "per_difficulty": {str(k): v for k, v in self.per_difficulty.items()},
            "symcode_results": [_solve_result_to_dict(r) for r in self.symcode_results],
            "prose_results": [_solve_result_to_dict(r) for r in self.prose_results],
        }


def _solve_result_to_dict(r: SolveResult) -> dict[str, Any]:
    """Serialize a SolveResult to a JSON-compatible dict."""
    return {
        "problem": r.problem[:200],
        "expected_answer": r.expected_answer,
        "final_answer": r.final_answer,
        "correct": r.correct,
        "num_attempts": r.num_attempts,
        "pipeline": r.pipeline,
        "task_type": r.task_type,
        "total_time": r.total_time,
        "attempts": [
            {
                "attempt_number": a.attempt_number,
                "extracted_answer": a.extracted_answer,
                "answer_correct": a.answer_correct,
                "feedback": a.feedback[:200] if a.feedback else "",
                "error_type": a.execution_result.error.error_type
                if a.execution_result.error
                else None,
            }
            for a in r.attempts
        ],
    }


class BenchmarkRunner:
    """Run benchmark problems through SymCode and/or prose pipelines."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        generator: SymCodeGenerator | None = None,
        retry_loop: RetryLoop | None = None,
        prose_baseline: ProseBaseline | None = None,
        router: TaskRouter | None = None,
        checker: AnswerChecker | None = None,
    ):
        self.config = config or {}
        self.router = router or TaskRouter(config)
        self.checker = checker or AnswerChecker()

        if generator is None:
            generator = SymCodeGenerator(config=config, mock=True)
        self.generator = generator

        if retry_loop is None:
            retry_loop = RetryLoop(
                generator=self.generator,
                checker=self.checker,
                max_retries=self.config.get("verification", {}).get("max_retries", 3),
            )
        self.retry_loop = retry_loop

        if prose_baseline is None:
            prose_baseline = ProseBaseline(config=config, mock=True)
        self.prose_baseline = prose_baseline

    def run(
        self,
        problems: list[BenchmarkProblem],
        pipelines: list[str] | None = None,
        concurrency: int = 1,
    ) -> BenchmarkResult:
        """Run benchmark problems.

        Args:
            problems: List of BenchmarkProblem instances.
            pipelines: Which pipelines to run: ["symcode"], ["prose"], or
                       ["symcode", "prose"] (default: both).
            concurrency: Number of concurrent workers.

        Returns:
            BenchmarkResult with all results and summaries.
        """
        if pipelines is None:
            pipelines = ["symcode", "prose"]

        start = time.time()
        result = BenchmarkResult()

        if "symcode" in pipelines:
            result.symcode_results = self._run_pipeline(
                problems, "symcode", concurrency
            )

        if "prose" in pipelines:
            result.prose_results = self._run_pipeline(
                problems, "prose", concurrency
            )

        result.total_time = time.time() - start
        result.compute_summaries()
        return result

    def _run_pipeline(
        self,
        problems: list[BenchmarkProblem],
        pipeline: str,
        concurrency: int,
    ) -> list[SolveResult]:
        """Run a single pipeline on all problems."""
        logger.info("Running %s pipeline on %d problems", pipeline, len(problems))

        if concurrency <= 1:
            results = []
            for p in problems:
                results.append(self._solve_one(p, pipeline))
            return results

        results: list[SolveResult | None] = [None] * len(problems)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(self._solve_one, p, pipeline): i
                for i, p in enumerate(problems)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Problem %d failed: %s", idx, exc)
                    results[idx] = SolveResult(
                        problem=problems[idx].problem,
                        expected_answer=problems[idx].expected_answer,
                        correct=False,
                        pipeline=pipeline,
                    )

        return [r for r in results if r is not None]

    def _solve_one(self, problem: BenchmarkProblem, pipeline: str) -> SolveResult:
        """Solve a single problem with the given pipeline."""
        routing = self.router.heuristic_route(
            problem.problem, {"subject": problem.subject}
        )

        if pipeline == "symcode":
            solve_result = self.retry_loop.solve(
                problem=problem.problem,
                task_type=routing.task_type,
                expected_answer=problem.expected_answer,
            )
            solve_result.task_type = problem.subject or routing.task_type.value
            # Stash difficulty for breakdown computation
            object.__setattr__(solve_result, "_difficulty", problem.difficulty)
            return solve_result

        # prose pipeline
        answer_str, response = self.prose_baseline.solve(problem.problem)
        correct = False
        if answer_str is not None and problem.expected_answer:
            correct = self.checker.check(answer_str, problem.expected_answer)

        result = SolveResult(
            problem=problem.problem,
            expected_answer=problem.expected_answer,
            final_answer=answer_str,
            correct=correct,
            num_attempts=1,
            task_type=problem.subject or routing.task_type.value,
            pipeline="prose",
        )
        object.__setattr__(result, "_difficulty", problem.difficulty)
        return result

    def save_results(
        self,
        result: BenchmarkResult,
        output_path: str | Path,
    ) -> Path:
        """Save benchmark results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = result.to_dict()
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Results saved to %s", path)
        return path
