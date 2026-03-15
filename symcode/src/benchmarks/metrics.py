"""Metrics computation for benchmark evaluation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.verification.result_types import SolveResult
from src.utils.logging import get_logger

logger = get_logger("benchmarks.metrics")


class MetricsComputer:
    """Compute evaluation metrics from solve results."""

    # ── accuracy ────────────────────────────────────────────────────

    @staticmethod
    def accuracy(results: list[SolveResult]) -> float:
        """Compute simple accuracy (fraction correct)."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.correct) / len(results)

    # ── pass@k (unbiased estimator) ─────────────────────────────────

    @staticmethod
    def pass_at_k(
        n: int,
        c: int,
        k: int,
    ) -> float:
        """Unbiased estimator for pass@k.

        Args:
            n: Total number of samples generated.
            c: Number of correct samples.
            k: k in pass@k.

        Returns:
            Estimated pass@k probability.

        Uses the formula: 1 - C(n-c, k) / C(n, k)
        """
        if n < k:
            return 1.0 if c > 0 else 0.0
        if c == 0:
            return 0.0
        if n - c < k:
            return 1.0

        # Use log-space to avoid overflow
        # C(n-c, k) / C(n, k) = prod_{i=0}^{k-1} (n-c-i) / (n-i)
        log_ratio = 0.0
        for i in range(k):
            log_ratio += math.log(n - c - i) - math.log(n - i)
        return 1.0 - math.exp(log_ratio)

    @staticmethod
    def pass_at_k_from_results(
        results: list[SolveResult],
        k: int,
    ) -> float:
        """Compute pass@k from a list of SolveResults.

        Each result represents one problem. We use num_attempts as n
        and count correct attempts as c.
        """
        if not results:
            return 0.0

        pass_rates: list[float] = []
        for r in results:
            n = r.num_attempts
            c = sum(1 for a in r.attempts if a.answer_correct)
            pass_rates.append(MetricsComputer.pass_at_k(n, c, k))

        return sum(pass_rates) / len(pass_rates)

    # ── accuracy by subject ─────────────────────────────────────────

    @staticmethod
    def accuracy_by_subject(
        results: list[SolveResult],
    ) -> dict[str, dict[str, Any]]:
        """Compute accuracy broken down by subject/task_type.

        Returns:
            {subject: {"accuracy": float, "correct": int, "total": int}}
        """
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            subject = r.task_type or "unknown"
            groups[subject].append(r.correct)

        breakdown: dict[str, dict[str, Any]] = {}
        for subject, correct_list in sorted(groups.items()):
            total = len(correct_list)
            correct = sum(correct_list)
            breakdown[subject] = {
                "accuracy": correct / total if total else 0.0,
                "correct": correct,
                "total": total,
            }
        return breakdown

    # ── accuracy by difficulty ──────────────────────────────────────

    @staticmethod
    def accuracy_by_difficulty(
        results: list[SolveResult],
        difficulties: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Compute accuracy broken down by difficulty level.

        Args:
            results: Solve results.
            difficulties: Parallel list of difficulty values. If None,
                         attempts to read a ``_difficulty`` attribute
                         from each result.

        Returns:
            {difficulty: {"accuracy": float, "correct": int, "total": int}}
        """
        groups: dict[int, list[bool]] = defaultdict(list)
        for i, r in enumerate(results):
            if difficulties is not None and i < len(difficulties):
                diff = difficulties[i]
            else:
                diff = getattr(r, "_difficulty", 0)
            groups[diff].append(r.correct)

        breakdown: dict[int, dict[str, Any]] = {}
        for diff, correct_list in sorted(groups.items()):
            total = len(correct_list)
            correct = sum(correct_list)
            breakdown[diff] = {
                "accuracy": correct / total if total else 0.0,
                "correct": correct,
                "total": total,
            }
        return breakdown

    # ── retry effectiveness ─────────────────────────────────────────

    @staticmethod
    def retry_effectiveness(results: list[SolveResult]) -> dict[str, Any]:
        """Compute retry/self-correction effectiveness metrics.

        Returns:
            - first_attempt_accuracy: fraction correct on attempt 1
            - final_accuracy: fraction correct after all retries
            - recovery_rate: fraction of first-attempt failures recovered
            - avg_attempts_when_correct: average attempts for correct answers
            - avg_attempts_when_wrong: average attempts for wrong answers
        """
        if not results:
            return {
                "first_attempt_accuracy": 0.0,
                "final_accuracy": 0.0,
                "recovery_rate": 0.0,
                "avg_attempts_when_correct": 0.0,
                "avg_attempts_when_wrong": 0.0,
            }

        first_correct = 0
        final_correct = 0
        first_wrong_then_correct = 0
        first_wrong = 0
        attempts_correct: list[int] = []
        attempts_wrong: list[int] = []

        for r in results:
            # Check if first attempt was correct
            first_ok = (
                len(r.attempts) > 0
                and r.attempts[0].answer_correct is True
            )
            if first_ok:
                first_correct += 1
            else:
                first_wrong += 1

            if r.correct:
                final_correct += 1
                attempts_correct.append(r.num_attempts)
                if not first_ok:
                    first_wrong_then_correct += 1
            else:
                attempts_wrong.append(r.num_attempts)

        n = len(results)
        return {
            "first_attempt_accuracy": first_correct / n,
            "final_accuracy": final_correct / n,
            "recovery_rate": (
                first_wrong_then_correct / first_wrong if first_wrong > 0 else 0.0
            ),
            "avg_attempts_when_correct": (
                sum(attempts_correct) / len(attempts_correct)
                if attempts_correct
                else 0.0
            ),
            "avg_attempts_when_wrong": (
                sum(attempts_wrong) / len(attempts_wrong)
                if attempts_wrong
                else 0.0
            ),
        }

    # ── error distribution ──────────────────────────────────────────

    @staticmethod
    def error_distribution(
        results: list[SolveResult],
    ) -> dict[str, int]:
        """Count error types across all attempts.

        Returns:
            {error_type: count}
        """
        counter: Counter[str] = Counter()
        for r in results:
            for a in r.attempts:
                if a.execution_result.error is not None:
                    counter[a.execution_result.error.error_type] += 1
        return dict(counter.most_common())

    # ── token efficiency ────────────────────────────────────────────

    @staticmethod
    def token_efficiency(
        results: list[SolveResult],
    ) -> dict[str, float]:
        """Compute token usage efficiency metrics.

        Returns:
            - avg_attempts: average number of attempts per problem
            - total_attempts: total across all problems
            - attempts_per_correct: average attempts per correct answer
        """
        if not results:
            return {
                "avg_attempts": 0.0,
                "total_attempts": 0,
                "attempts_per_correct": 0.0,
            }

        total_attempts = sum(r.num_attempts for r in results)
        correct_count = sum(1 for r in results if r.correct)

        return {
            "avg_attempts": total_attempts / len(results),
            "total_attempts": total_attempts,
            "attempts_per_correct": (
                total_attempts / correct_count if correct_count > 0 else float("inf")
            ),
        }

    # ── latency stats ──────────────────────────────────────────────

    @staticmethod
    def latency_stats(
        results: list[SolveResult],
    ) -> dict[str, float]:
        """Compute latency statistics.

        Returns:
            - mean: average solve time
            - median: median solve time
            - p95: 95th percentile solve time
            - total: total wall-clock time
        """
        times = [r.total_time for r in results if r.total_time > 0]
        if not times:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "total": 0.0}

        times_sorted = sorted(times)
        n = len(times_sorted)
        p95_idx = min(int(n * 0.95), n - 1)

        return {
            "mean": sum(times_sorted) / n,
            "median": times_sorted[n // 2],
            "p95": times_sorted[p95_idx],
            "total": sum(times_sorted),
        }
