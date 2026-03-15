"""Head-to-head comparison analysis: SymCode vs prose baseline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.benchmarks.metrics import MetricsComputer
from src.verification.result_types import SolveResult
from src.utils.logging import get_logger

logger = get_logger("benchmarks.comparison")


class ComparisonAnalyzer:
    """Analyze head-to-head comparison between SymCode and prose pipelines."""

    def __init__(
        self,
        symcode_results: list[SolveResult],
        prose_results: list[SolveResult],
    ):
        if len(symcode_results) != len(prose_results):
            raise ValueError(
                f"Result lists must have same length: "
                f"{len(symcode_results)} vs {len(prose_results)}"
            )
        self.symcode = symcode_results
        self.prose = prose_results
        self.n = len(symcode_results)

    # ── summary ─────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Compute overall comparison summary.

        Includes accuracy for both, delta, and McNemar's test p-value
        for statistical significance.
        """
        sym_acc = MetricsComputer.accuracy(self.symcode)
        prose_acc = MetricsComputer.accuracy(self.prose)
        delta = sym_acc - prose_acc

        # Contingency counts for McNemar's test
        b, c = self._mcnemar_counts()
        p_value = self._mcnemar_test(b, c)

        sym_only = sum(
            1 for s, p in zip(self.symcode, self.prose)
            if s.correct and not p.correct
        )
        prose_only = sum(
            1 for s, p in zip(self.symcode, self.prose)
            if not s.correct and p.correct
        )
        both_correct = sum(
            1 for s, p in zip(self.symcode, self.prose)
            if s.correct and p.correct
        )
        both_wrong = sum(
            1 for s, p in zip(self.symcode, self.prose)
            if not s.correct and not p.correct
        )

        return {
            "symcode_accuracy": sym_acc,
            "prose_accuracy": prose_acc,
            "delta_pp": delta * 100,  # percentage points
            "mcnemar_p_value": p_value,
            "significant": bool(p_value < 0.05) if p_value is not None else None,
            "symcode_only_correct": sym_only,
            "prose_only_correct": prose_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "n": self.n,
        }

    # ── per-subject comparison ──────────────────────────────────────

    def per_subject_comparison(self) -> dict[str, dict[str, Any]]:
        """Compare accuracy by subject/task_type."""
        groups: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: {"symcode": [], "prose": []}
        )
        for s, p in zip(self.symcode, self.prose):
            subject = s.task_type or "unknown"
            groups[subject]["symcode"].append(s.correct)
            groups[subject]["prose"].append(p.correct)

        result: dict[str, dict[str, Any]] = {}
        for subject, data in sorted(groups.items()):
            sym_acc = sum(data["symcode"]) / len(data["symcode"]) if data["symcode"] else 0.0
            prose_acc = sum(data["prose"]) / len(data["prose"]) if data["prose"] else 0.0
            result[subject] = {
                "symcode_accuracy": sym_acc,
                "prose_accuracy": prose_acc,
                "delta_pp": (sym_acc - prose_acc) * 100,
                "count": len(data["symcode"]),
            }
        return result

    # ── per-difficulty comparison ───────────────────────────────────

    def per_difficulty_comparison(
        self,
        difficulties: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Compare accuracy by difficulty level.

        Args:
            difficulties: Parallel list of difficulty values. If None,
                         reads ``_difficulty`` attribute from results.
        """
        groups: dict[int, dict[str, list[bool]]] = defaultdict(
            lambda: {"symcode": [], "prose": []}
        )
        for i, (s, p) in enumerate(zip(self.symcode, self.prose)):
            if difficulties is not None and i < len(difficulties):
                diff = difficulties[i]
            else:
                diff = getattr(s, "_difficulty", 0)
            groups[diff]["symcode"].append(s.correct)
            groups[diff]["prose"].append(p.correct)

        result: dict[int, dict[str, Any]] = {}
        for diff, data in sorted(groups.items()):
            sym_acc = sum(data["symcode"]) / len(data["symcode"]) if data["symcode"] else 0.0
            prose_acc = sum(data["prose"]) / len(data["prose"]) if data["prose"] else 0.0
            result[diff] = {
                "symcode_accuracy": sym_acc,
                "prose_accuracy": prose_acc,
                "delta_pp": (sym_acc - prose_acc) * 100,
                "count": len(data["symcode"]),
            }
        return result

    # ── failure mode comparison ─────────────────────────────────────

    def failure_mode_comparison(self) -> dict[str, Any]:
        """Analyze failure modes for each pipeline.

        Returns error distributions and categorized failure counts.
        """
        sym_errors = MetricsComputer.error_distribution(self.symcode)
        prose_errors = MetricsComputer.error_distribution(self.prose)

        # Categorize failures
        sym_failures: dict[str, int] = defaultdict(int)
        prose_failures: dict[str, int] = defaultdict(int)

        for r in self.symcode:
            if not r.correct:
                if r.final_answer is None:
                    sym_failures["no_answer"] += 1
                elif any(a.execution_result.error for a in r.attempts):
                    sym_failures["execution_error"] += 1
                else:
                    sym_failures["wrong_answer"] += 1

        for r in self.prose:
            if not r.correct:
                if r.final_answer is None:
                    prose_failures["no_answer"] += 1
                else:
                    prose_failures["wrong_answer"] += 1

        return {
            "symcode_error_distribution": sym_errors,
            "prose_error_distribution": prose_errors,
            "symcode_failure_modes": dict(sym_failures),
            "prose_failure_modes": dict(prose_failures),
        }

    # ── qualitative examples ────────────────────────────────────────

    def qualitative_examples(
        self, max_examples: int = 5
    ) -> dict[str, list[dict[str, Any]]]:
        """Select qualitative examples for analysis.

        Returns:
            - symcode_wins: problems where SymCode correct, prose wrong
            - prose_wins: problems where prose correct, SymCode wrong
            - both_fail: problems where both pipelines failed
        """
        symcode_wins: list[dict[str, Any]] = []
        prose_wins: list[dict[str, Any]] = []
        both_fail: list[dict[str, Any]] = []

        for i, (s, p) in enumerate(zip(self.symcode, self.prose)):
            entry = {
                "problem_index": i,
                "problem": s.problem[:300],
                "expected_answer": s.expected_answer,
                "symcode_answer": s.final_answer,
                "prose_answer": p.final_answer,
                "symcode_attempts": s.num_attempts,
                "subject": s.task_type,
            }

            if s.correct and not p.correct:
                if len(symcode_wins) < max_examples:
                    symcode_wins.append(entry)
            elif not s.correct and p.correct:
                if len(prose_wins) < max_examples:
                    prose_wins.append(entry)
            elif not s.correct and not p.correct:
                if len(both_fail) < max_examples:
                    both_fail.append(entry)

        return {
            "symcode_wins": symcode_wins,
            "prose_wins": prose_wins,
            "both_fail": both_fail,
        }

    # ── McNemar's statistical test ──────────────────────────────────

    def statistical_test(self) -> dict[str, Any]:
        """Run McNemar's test for paired comparisons.

        Returns:
            - b: SymCode correct, prose wrong count
            - c: prose correct, SymCode wrong count
            - chi2: test statistic
            - p_value: p-value
            - significant_005: whether significant at alpha=0.05
        """
        b, c = self._mcnemar_counts()
        p_value = self._mcnemar_test(b, c)

        chi2 = None
        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0

        return {
            "b_symcode_only": b,
            "c_prose_only": c,
            "chi2": chi2,
            "p_value": p_value,
            "significant_005": bool(p_value < 0.05) if p_value is not None else None,
        }

    def _mcnemar_counts(self) -> tuple[int, int]:
        """Count discordant pairs for McNemar's test.

        b = SymCode correct & prose wrong
        c = prose correct & SymCode wrong
        """
        b = 0
        c = 0
        for s, p in zip(self.symcode, self.prose):
            if s.correct and not p.correct:
                b += 1
            elif not s.correct and p.correct:
                c += 1
        return b, c

    @staticmethod
    def _mcnemar_test(b: int, c: int) -> float | None:
        """Compute McNemar's test p-value.

        Uses the chi-squared approximation with continuity correction.
        Falls back to scipy.stats if available, otherwise uses a
        simple approximation.
        """
        if b + c == 0:
            return 1.0

        # Chi-squared statistic with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)

        # Try scipy for exact p-value
        try:
            from scipy.stats import chi2 as chi2_dist

            p_value = 1.0 - chi2_dist.cdf(chi2, df=1)
            return p_value
        except ImportError:
            pass

        # Fallback: rough approximation using normal CDF
        # P(chi2 > x) for df=1 ~ 2 * (1 - Phi(sqrt(x)))
        import math

        z = math.sqrt(chi2)
        # Simple normal CDF approximation (Abramowitz & Stegun)
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        cdf = 1.0 - d * math.exp(-0.5 * z * z) * poly
        p_value = 2.0 * (1.0 - cdf)
        return max(0.0, min(1.0, p_value))
