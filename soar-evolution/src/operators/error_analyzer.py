"""Error analysis producing structured mutation hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.arc.evaluator import ProgramEvalResult, ProgramEvaluator
from src.arc.grid import ARCTask, Grid
from src.arc.visualizer import GridVisualizer
from src.population.individual import Individual


@dataclass
class ErrorAnalysis:
    """Structured analysis of program errors."""

    error_type: str  # "compile", "runtime", "wrong_output", "shape_mismatch"
    severity: float  # 0.0 to 1.0
    description: str
    hints: List[str] = field(default_factory=list)
    affected_examples: List[int] = field(default_factory=list)
    common_patterns: List[str] = field(default_factory=list)


class ErrorAnalyzer:
    """Analyzes program errors to produce actionable mutation hints."""

    def __init__(
        self,
        evaluator: Optional[ProgramEvaluator] = None,
        visualizer: Optional[GridVisualizer] = None,
    ):
        self.evaluator = evaluator or ProgramEvaluator()
        self.visualizer = visualizer or GridVisualizer()

    def analyze(
        self,
        individual: Individual,
        task: ARCTask,
    ) -> List[ErrorAnalysis]:
        """Analyze all errors in an individual's program."""
        analyses = []

        eval_result = self.evaluator.evaluate_task(individual.code, task)

        # Check for compile errors
        if eval_result.compile_error:
            analyses.append(
                ErrorAnalysis(
                    error_type="compile",
                    severity=1.0,
                    description=eval_result.compile_error,
                    hints=[
                        "Fix syntax errors",
                        "Ensure 'def transform(input_grid):' is defined",
                        "Check for missing imports or undefined variables",
                    ],
                )
            )
            return analyses

        # Analyze training results
        for i, result in enumerate(eval_result.train_results):
            if result.error:
                analysis = self._analyze_runtime_error(result.error, i)
                analyses.append(analysis)
            elif result.output_grid and result.expected_grid:
                if result.output_grid.shape != result.expected_grid.shape:
                    analyses.append(
                        ErrorAnalysis(
                            error_type="shape_mismatch",
                            severity=0.8,
                            description=(
                                f"Example {i}: output shape {result.output_grid.shape} "
                                f"!= expected {result.expected_grid.shape}"
                            ),
                            hints=[
                                "Check output grid dimensions",
                                "The transform may need to resize the grid",
                            ],
                            affected_examples=[i],
                        )
                    )
                elif not result.correct:
                    analyses.append(
                        self._analyze_wrong_output(
                            result.output_grid,
                            result.expected_grid,
                            i,
                        )
                    )

        # Find common patterns across errors
        if analyses:
            self._find_common_patterns(analyses)

        return analyses

    def _analyze_runtime_error(
        self, error: str, example_idx: int
    ) -> ErrorAnalysis:
        """Analyze a runtime error."""
        hints = []

        if "IndexError" in error:
            hints.append("Check array bounds and grid dimensions")
            hints.append("Ensure row/column indices are within range")
        elif "TypeError" in error:
            hints.append("Check data types - grid cells are integers")
            hints.append("Ensure function returns a list of lists")
        elif "KeyError" in error:
            hints.append("Check dictionary key access")
        elif "ZeroDivisionError" in error:
            hints.append("Add division-by-zero checks")
        else:
            hints.append("Review the logic for edge cases")

        return ErrorAnalysis(
            error_type="runtime",
            severity=0.9,
            description=error,
            hints=hints,
            affected_examples=[example_idx],
        )

    def _analyze_wrong_output(
        self,
        output: Grid,
        expected: Grid,
        example_idx: int,
    ) -> ErrorAnalysis:
        """Analyze incorrect output."""
        hints = []

        # Count differences
        diff_count = 0
        total = expected.pixel_count()
        for r in range(expected.height):
            for c in range(expected.width):
                if output.data[r][c] != expected.data[r][c]:
                    diff_count += 1

        ratio = diff_count / max(total, 1)

        if ratio > 0.8:
            hints.append("Output is very different - reconsider the approach")
            hints.append("The transformation logic may be fundamentally wrong")
        elif ratio > 0.3:
            hints.append("Many pixels wrong - check the core transformation")
        else:
            hints.append("Close to correct - fine-tune the edge cases")
            hints.append("Check specific pixel positions that differ")

        # Check for systematic errors
        output_colors = output.colors_used()
        expected_colors = expected.colors_used()

        if output_colors != expected_colors:
            missing = expected_colors - output_colors
            extra = output_colors - expected_colors
            if missing:
                hints.append(f"Missing colors in output: {missing}")
            if extra:
                hints.append(f"Extra colors in output: {extra}")

        return ErrorAnalysis(
            error_type="wrong_output",
            severity=ratio,
            description=(
                f"Example {example_idx}: {diff_count}/{total} pixels wrong "
                f"({ratio:.1%})"
            ),
            hints=hints,
            affected_examples=[example_idx],
        )

    def _find_common_patterns(self, analyses: List[ErrorAnalysis]) -> None:
        """Find common patterns across multiple error analyses."""
        error_types = [a.error_type for a in analyses]

        if error_types.count("runtime") > 1:
            for a in analyses:
                if "runtime" in a.error_type:
                    a.common_patterns.append(
                        "Multiple runtime errors suggest structural issues"
                    )

        if error_types.count("wrong_output") > 1:
            for a in analyses:
                if a.error_type == "wrong_output":
                    a.common_patterns.append(
                        "Multiple wrong outputs suggest the core logic needs revision"
                    )

    def get_mutation_hints(
        self,
        individual: Individual,
        task: ARCTask,
    ) -> Dict:
        """Get structured hints for guiding mutations."""
        analyses = self.analyze(individual, task)

        if not analyses:
            return {
                "suggested_mutation": "REFINEMENT",
                "hints": ["Program appears to work - try refinement"],
                "severity": 0.0,
            }

        # Determine suggested mutation type
        max_severity = max(a.severity for a in analyses)
        error_types = set(a.error_type for a in analyses)

        if "compile" in error_types:
            suggested = "BUG_FIX"
        elif "runtime" in error_types:
            suggested = "BUG_FIX"
        elif max_severity > 0.5:
            suggested = "RESTRUCTURE"
        else:
            suggested = "REFINEMENT"

        all_hints = []
        for a in analyses:
            all_hints.extend(a.hints)

        return {
            "suggested_mutation": suggested,
            "hints": all_hints[:10],
            "severity": max_severity,
            "error_count": len(analyses),
            "error_types": list(error_types),
        }
