"""Retry / self-correction analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.verification.result_types import SolveResult
from src.utils.logging import get_logger

logger = get_logger("analysis.retry_analysis")


class RetryAnalyzer:
    """Analyze the effectiveness of the retry / self-correction loop."""

    # ── correction success by error type ────────────────────────────

    @staticmethod
    def correction_success_by_error_type(
        results: list[SolveResult],
    ) -> dict[str, dict[str, Any]]:
        """Compute self-correction success rate by error type.

        For each error type, counts how often the next attempt
        succeeded after seeing that error.

        Returns:
            {error_type: {
                "attempts": int,
                "successes": int,
                "rate": float,
            }}
        """
        stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0}
        )

        for r in results:
            for i in range(len(r.attempts) - 1):
                attempt = r.attempts[i]
                next_attempt = r.attempts[i + 1]

                error = attempt.execution_result.error
                if error is None:
                    # Could be wrong-answer case
                    if attempt.answer_correct is False:
                        error_type = "WrongAnswer"
                    else:
                        continue
                else:
                    error_type = error.error_type

                stats[error_type]["attempts"] += 1
                # Success = next attempt either ran correctly or produced correct answer
                if (
                    next_attempt.execution_result.success
                    and next_attempt.answer_correct is not False
                ):
                    stats[error_type]["successes"] += 1

        result: dict[str, dict[str, Any]] = {}
        for error_type, data in sorted(stats.items()):
            result[error_type] = {
                "attempts": data["attempts"],
                "successes": data["successes"],
                "rate": data["successes"] / data["attempts"] if data["attempts"] > 0 else 0.0,
            }
        return result

    # ── correction trajectory ───────────────────────────────────────

    @staticmethod
    def correction_trajectory(
        results: list[SolveResult],
        max_k: int = 5,
    ) -> dict[str, Any]:
        """Compute cumulative accuracy at each attempt k.

        Shows how accuracy improves with each retry.

        Returns:
            {
                "cumulative_accuracy": [acc_at_k1, acc_at_k2, ...],
                "marginal_gain": [0, gain_at_k2, gain_at_k3, ...],
                "problems_solved_at_k": [count_k1, count_k2, ...],
            }
        """
        if not results:
            return {
                "cumulative_accuracy": [],
                "marginal_gain": [],
                "problems_solved_at_k": [],
            }

        n = len(results)
        cumulative: list[float] = []
        solved_at: list[int] = []

        for k in range(1, max_k + 1):
            solved = 0
            for r in results:
                # Check if any of the first k attempts was correct
                for a in r.attempts[:k]:
                    if a.answer_correct is True:
                        solved += 1
                        break
            cumulative.append(solved / n)
            solved_at.append(solved)

        marginal = [0.0] + [
            cumulative[i] - cumulative[i - 1]
            for i in range(1, len(cumulative))
        ]

        return {
            "cumulative_accuracy": cumulative,
            "marginal_gain": marginal,
            "problems_solved_at_k": solved_at,
        }

    # ── optimal k ───────────────────────────────────────────────────

    @staticmethod
    def optimal_k(
        results: list[SolveResult],
        max_k: int = 10,
        cost_per_attempt: float = 1.0,
    ) -> dict[str, Any]:
        """Find the optimal number of retries balancing accuracy vs cost.

        Uses a simple utility function: utility = accuracy - cost_weight * k

        Args:
            results: Solve results.
            max_k: Maximum k to consider.
            cost_per_attempt: Cost weight per additional attempt.

        Returns:
            - optimal_k: best k value
            - accuracy_at_optimal: accuracy at that k
            - analysis: list of {k, accuracy, marginal_gain, utility}
        """
        if not results:
            return {"optimal_k": 1, "accuracy_at_optimal": 0.0, "analysis": []}

        n = len(results)
        analysis: list[dict[str, Any]] = []
        best_k = 1
        best_utility = -float("inf")

        for k in range(1, max_k + 1):
            solved = 0
            for r in results:
                for a in r.attempts[:k]:
                    if a.answer_correct is True:
                        solved += 1
                        break
            accuracy = solved / n

            # Marginal gain
            prev_acc = analysis[-1]["accuracy"] if analysis else 0.0
            marginal = accuracy - prev_acc

            # Utility: accuracy gain minus cost
            utility = accuracy - cost_per_attempt * k / n

            analysis.append({
                "k": k,
                "accuracy": accuracy,
                "marginal_gain": marginal,
                "utility": utility,
            })

            if utility > best_utility:
                best_utility = utility
                best_k = k

        return {
            "optimal_k": best_k,
            "accuracy_at_optimal": next(
                (a["accuracy"] for a in analysis if a["k"] == best_k), 0.0
            ),
            "analysis": analysis,
        }

    # ── retry funnel plot ───────────────────────────────────────────

    @staticmethod
    def plot_retry_funnel(
        results: list[SolveResult],
        output_path: str | None = None,
        max_k: int = 5,
    ) -> Any:
        """Plot a retry funnel showing problems solved at each attempt.

        Returns the matplotlib figure, or None if matplotlib is unavailable.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return None

        if not results:
            return None

        n = len(results)
        # Count first-solve-at-k
        first_solve: Counter[int] = Counter()
        for r in results:
            for a in r.attempts:
                if a.answer_correct is True:
                    first_solve[a.attempt_number] += 1
                    break

        unsolved = n - sum(first_solve.values())

        ks = list(range(1, max_k + 1))
        counts = [first_solve.get(k, 0) for k in ks]
        counts.append(unsolved)
        labels = [f"Attempt {k}" for k in ks] + ["Unsolved"]
        colors = ["#2ecc71"] * len(ks) + ["#e74c3c"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)

        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_ylabel("Number of Problems")
        ax.set_title("Retry Funnel: When Problems Are First Solved")
        ax.grid(axis="y", alpha=0.3)

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Retry funnel saved to %s", output_path)

        return fig
