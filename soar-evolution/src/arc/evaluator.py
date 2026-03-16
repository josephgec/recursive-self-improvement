"""Program evaluator for ARC tasks - executes transform functions safely."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.arc.grid import ARCExample, ARCTask, Grid


@dataclass
class EvalResult:
    """Result of evaluating a program on a single example."""

    success: bool
    output_grid: Optional[Grid] = None
    expected_grid: Optional[Grid] = None
    error: Optional[str] = None
    pixel_accuracy: float = 0.0
    correct: bool = False

    def __post_init__(self):
        if self.success and self.output_grid and self.expected_grid:
            self.pixel_accuracy = compute_pixel_accuracy(
                self.output_grid, self.expected_grid
            )
            self.correct = self.pixel_accuracy == 1.0


@dataclass
class ProgramEvalResult:
    """Result of evaluating a program on an entire task."""

    train_results: List[EvalResult] = field(default_factory=list)
    test_results: List[EvalResult] = field(default_factory=list)
    compile_error: Optional[str] = None

    @property
    def train_accuracy(self) -> float:
        if not self.train_results:
            return 0.0
        return sum(r.pixel_accuracy for r in self.train_results) / len(
            self.train_results
        )

    @property
    def test_accuracy(self) -> float:
        if not self.test_results:
            return 0.0
        return sum(r.pixel_accuracy for r in self.test_results) / len(
            self.test_results
        )

    @property
    def all_train_correct(self) -> bool:
        return all(r.correct for r in self.train_results) if self.train_results else False

    @property
    def all_test_correct(self) -> bool:
        return all(r.correct for r in self.test_results) if self.test_results else False

    @property
    def fully_correct(self) -> bool:
        return self.all_train_correct and self.all_test_correct


def compute_pixel_accuracy(output: Grid, expected: Grid) -> float:
    """Compute pixel-level accuracy between two grids."""
    if output.shape != expected.shape:
        return 0.0

    total = output.pixel_count()
    if total == 0:
        return 1.0

    correct = 0
    for r in range(output.height):
        for c in range(output.width):
            if output.data[r][c] == expected.data[r][c]:
                correct += 1

    return correct / total


class ProgramEvaluator:
    """Evaluates transform programs against ARC tasks."""

    def __init__(self, timeout_seconds: float = 5.0, max_retries: int = 1):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def compile_program(self, code: str) -> Tuple[bool, Optional[Callable], Optional[str]]:
        """Compile program code and extract the transform function."""
        namespace: Dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return False, None, f"Compilation error: {type(e).__name__}: {e}"

        transform = namespace.get("transform")
        if transform is None:
            return False, None, "No 'transform' function defined"

        if not callable(transform):
            return False, None, "'transform' is not callable"

        return True, transform, None

    def evaluate_example(
        self,
        transform: Callable,
        example: ARCExample,
    ) -> EvalResult:
        """Evaluate a transform function on a single example."""
        try:
            input_data = example.input_grid.to_list()
            result = transform(input_data)

            if result is None:
                return EvalResult(
                    success=False,
                    expected_grid=example.output_grid,
                    error="Transform returned None",
                )

            if not isinstance(result, list):
                return EvalResult(
                    success=False,
                    expected_grid=example.output_grid,
                    error=f"Transform returned {type(result).__name__}, expected list",
                )

            output_grid = Grid.from_list(result)
            return EvalResult(
                success=True,
                output_grid=output_grid,
                expected_grid=example.output_grid,
            )

        except Exception as e:
            return EvalResult(
                success=False,
                expected_grid=example.output_grid,
                error=f"Runtime error: {type(e).__name__}: {e}",
            )

    def evaluate_task(
        self,
        code: str,
        task: ARCTask,
        eval_test: bool = True,
    ) -> ProgramEvalResult:
        """Evaluate a program on an entire ARC task."""
        result = ProgramEvalResult()

        ok, transform, error = self.compile_program(code)
        if not ok:
            result.compile_error = error
            return result

        # Evaluate on training examples
        for example in task.train:
            eval_result = self.evaluate_example(transform, example)
            result.train_results.append(eval_result)

        # Evaluate on test examples
        if eval_test:
            for example in task.test:
                eval_result = self.evaluate_example(transform, example)
                result.test_results.append(eval_result)

        return result

    def quick_check(self, code: str, task: ARCTask) -> float:
        """Quick check returning train accuracy only."""
        result = self.evaluate_task(code, task, eval_test=False)
        return result.train_accuracy
