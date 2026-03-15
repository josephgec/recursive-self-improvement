"""Error taxonomy analysis for SymCode pipeline failures."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.verification.result_types import SolveResult
from src.utils.logging import get_logger

logger = get_logger("analysis.error_taxonomy")


# ── Taxonomy categories ─────────────────────────────────────────────

TAXONOMY = {
    "code_generation_failures": {
        "description": "Failures in LLM code generation",
        "subcategories": {
            "no_code_generated": "LLM returned empty or unparseable code",
            "syntax_error": "Generated code has syntax errors",
            "import_error": "Missing or incorrect imports",
            "wrong_api_usage": "Incorrect use of SymPy or library APIs",
        },
    },
    "runtime_failures": {
        "description": "Failures during code execution",
        "subcategories": {
            "name_error": "Undefined variables or functions",
            "type_error": "Type mismatches in operations",
            "division_by_zero": "Division by zero errors",
            "timeout": "Code exceeded time limit",
            "memory_error": "Code exceeded memory limit",
            "other_runtime": "Other runtime errors",
        },
    },
    "logic_failures": {
        "description": "Code runs but produces wrong answer",
        "subcategories": {
            "wrong_formula": "Incorrect mathematical formula or approach",
            "off_by_one": "Off-by-one or boundary errors",
            "misinterpretation": "Problem misunderstood by the model",
            "no_answer_extracted": "Code ran but no answer variable found",
        },
    },
    "self_correction_failures": {
        "description": "Failures in the retry/self-correction loop",
        "subcategories": {
            "repeated_same_error": "Same error type on consecutive attempts",
            "new_error_introduced": "Fix introduced a different error",
            "wrong_answer_persists": "Answer wrong across all retries",
            "degraded_answer": "Answer got worse after correction",
        },
    },
}


@dataclass
class TaxonomyReport:
    """Result of error taxonomy analysis."""

    # category -> subcategory -> count
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    # category -> total count
    category_totals: dict[str, int] = field(default_factory=dict)

    # Total number of problems analysed
    total_problems: int = 0

    # Total number of failed problems
    total_failures: int = 0

    # Representative examples per category (max 3 each)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "counts": self.counts,
            "category_totals": self.category_totals,
            "total_problems": self.total_problems,
            "total_failures": self.total_failures,
            "examples": self.examples,
        }


class ErrorTaxonomist:
    """Classify errors into a detailed taxonomy."""

    # Map Python error types to taxonomy subcategories
    _ERROR_MAP: dict[str, tuple[str, str]] = {
        "SyntaxError": ("code_generation_failures", "syntax_error"),
        "IndentationError": ("code_generation_failures", "syntax_error"),
        "TabError": ("code_generation_failures", "syntax_error"),
        "ImportError": ("code_generation_failures", "import_error"),
        "ModuleNotFoundError": ("code_generation_failures", "import_error"),
        "NameError": ("runtime_failures", "name_error"),
        "TypeError": ("runtime_failures", "type_error"),
        "ZeroDivisionError": ("runtime_failures", "division_by_zero"),
        "TimeoutError": ("runtime_failures", "timeout"),
        "MemoryError": ("runtime_failures", "memory_error"),
        "AttributeError": ("code_generation_failures", "wrong_api_usage"),
        "ValueError": ("runtime_failures", "other_runtime"),
        "IndexError": ("runtime_failures", "other_runtime"),
        "KeyError": ("runtime_failures", "other_runtime"),
        "RecursionError": ("runtime_failures", "other_runtime"),
        "OverflowError": ("runtime_failures", "other_runtime"),
        "RuntimeError": ("runtime_failures", "other_runtime"),
    }

    def analyze(self, results: list[SolveResult]) -> TaxonomyReport:
        """Analyze a list of solve results and produce a taxonomy report.

        Args:
            results: List of SolveResult from the SymCode pipeline.

        Returns:
            TaxonomyReport with categorized error counts.
        """
        report = TaxonomyReport(
            total_problems=len(results),
            total_failures=sum(1 for r in results if not r.correct),
        )

        # Initialize counts
        for cat, info in TAXONOMY.items():
            report.counts[cat] = {sub: 0 for sub in info["subcategories"]}
            report.category_totals[cat] = 0
            report.examples[cat] = []

        for r in results:
            if r.correct:
                continue

            categories_found: set[str] = set()

            # Analyze each attempt
            for i, attempt in enumerate(r.attempts):
                error = attempt.execution_result.error

                if error is not None:
                    cat, subcat = self._classify_error(error.error_type)
                    report.counts[cat][subcat] += 1
                    categories_found.add(cat)

                elif attempt.execution_result.success and attempt.extracted_answer is None:
                    report.counts["logic_failures"]["no_answer_extracted"] += 1
                    categories_found.add("logic_failures")

                elif (
                    attempt.execution_result.success
                    and attempt.answer_correct is False
                ):
                    report.counts["logic_failures"]["wrong_formula"] += 1
                    categories_found.add("logic_failures")

                # Check for no code generated
                if not attempt.code.strip():
                    report.counts["code_generation_failures"]["no_code_generated"] += 1
                    categories_found.add("code_generation_failures")

            # Self-correction analysis (look at attempt sequences)
            self._analyze_self_correction(r, report)

            # Update category totals
            for cat in categories_found:
                report.category_totals[cat] += 1

            # Store examples (max 3 per category)
            for cat in categories_found:
                if len(report.examples[cat]) < 3:
                    report.examples[cat].append({
                        "problem": r.problem[:200],
                        "expected_answer": r.expected_answer,
                        "final_answer": r.final_answer,
                        "num_attempts": r.num_attempts,
                    })

        return report

    def _classify_error(self, error_type: str) -> tuple[str, str]:
        """Map an error type string to (category, subcategory)."""
        if error_type in self._ERROR_MAP:
            return self._ERROR_MAP[error_type]
        return ("runtime_failures", "other_runtime")

    def _analyze_self_correction(
        self,
        result: SolveResult,
        report: TaxonomyReport,
    ) -> None:
        """Analyze self-correction patterns across attempts."""
        if len(result.attempts) < 2:
            return

        for i in range(1, len(result.attempts)):
            prev = result.attempts[i - 1]
            curr = result.attempts[i]

            prev_err = prev.execution_result.error
            curr_err = curr.execution_result.error

            # Same error type repeated
            if (
                prev_err is not None
                and curr_err is not None
                and prev_err.error_type == curr_err.error_type
            ):
                report.counts["self_correction_failures"]["repeated_same_error"] += 1

            # New error introduced (had error, fixed it, got different error)
            elif (
                prev_err is not None
                and curr_err is not None
                and prev_err.error_type != curr_err.error_type
            ):
                report.counts["self_correction_failures"]["new_error_introduced"] += 1

            # Answer was wrong and stayed wrong
            if (
                prev.answer_correct is False
                and curr.answer_correct is False
            ):
                report.counts["self_correction_failures"]["wrong_answer_persists"] += 1

            # Answer degraded (had correct-ish, now wrong)
            if (
                prev.answer_correct is True
                and curr.answer_correct is False
            ):
                report.counts["self_correction_failures"]["degraded_answer"] += 1

        # If last attempt is still wrong after retries, mark overall failure
        if not result.correct and len(result.attempts) > 1:
            report.category_totals["self_correction_failures"] = (
                report.category_totals.get("self_correction_failures", 0) + 1
            )
