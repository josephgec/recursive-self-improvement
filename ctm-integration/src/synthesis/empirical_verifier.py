"""Empirical verification of candidate rules against I/O examples."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.synthesis.candidate_generator import CandidateRule, IOExample


@dataclass
class FailedExample:
    """Details of a failed verification example."""

    input: Any
    expected_output: Any
    actual_output: Any
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of verifying a candidate rule."""

    rule_id: str
    passed: int
    total: int
    accuracy: float
    failures: List[FailedExample] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_perfect(self) -> bool:
        return self.passed == self.total and self.total > 0

    @property
    def is_partial(self) -> bool:
        return 0 < self.passed < self.total


class EmpiricalVerifier:
    """Verifies candidate rules by executing them against I/O examples.

    Rules are executed using exec() in an isolated namespace.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        """Initialize the verifier.

        Args:
            timeout: Maximum execution time per example (seconds).
                     Note: actual timeout enforcement requires subprocess;
                     here we use a simple try/except approach.
        """
        self.timeout = timeout

    def verify(
        self,
        rule: CandidateRule,
        examples: List[IOExample],
    ) -> VerificationResult:
        """Verify a rule against all examples.

        Args:
            rule: The candidate rule to verify.
            examples: List of I/O examples.

        Returns:
            VerificationResult with accuracy and failure details.
        """
        if not examples:
            return VerificationResult(
                rule_id=rule.rule_id, passed=0, total=0, accuracy=0.0
            )

        # Compile the rule
        func = self._compile_rule(rule)
        if func is None:
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=0,
                total=len(examples),
                accuracy=0.0,
                error="Failed to compile rule",
            )

        passed = 0
        failures = []

        for example in examples:
            try:
                result = self._execute(func, example.input)
                if self._outputs_match(result, example.output):
                    passed += 1
                else:
                    failures.append(
                        FailedExample(
                            input=example.input,
                            expected_output=example.output,
                            actual_output=result,
                        )
                    )
            except Exception as e:
                failures.append(
                    FailedExample(
                        input=example.input,
                        expected_output=example.output,
                        actual_output=None,
                        error=str(e),
                    )
                )

        accuracy = passed / len(examples)

        return VerificationResult(
            rule_id=rule.rule_id,
            passed=passed,
            total=len(examples),
            accuracy=accuracy,
            failures=failures,
        )

    def verify_generalization(
        self,
        rule: CandidateRule,
        train_examples: List[IOExample],
        test_examples: List[IOExample],
    ) -> dict:
        """Verify a rule on both training and test examples.

        Args:
            rule: The candidate rule.
            train_examples: Training I/O examples.
            test_examples: Held-out test examples.

        Returns:
            Dictionary with train and test VerificationResults.
        """
        train_result = self.verify(rule, train_examples)
        test_result = self.verify(rule, test_examples)

        return {
            "train": train_result,
            "test": test_result,
            "generalizes": test_result.accuracy >= train_result.accuracy * 0.8,
        }

    def _compile_rule(self, rule: CandidateRule) -> Optional[Any]:
        """Compile a rule and extract its callable function."""
        try:
            namespace: dict = {}
            exec(rule.source_code, namespace)
            # Find the first callable that isn't a builtin
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    return obj
            return None
        except Exception:
            return None

    def _execute(self, func: Any, input_val: Any) -> Any:
        """Execute a function with the given input.

        Args:
            func: Callable to execute.
            input_val: Input value.

        Returns:
            Function result.
        """
        if isinstance(input_val, (list, tuple)):
            # Try passing as single argument first
            try:
                return func(input_val)
            except TypeError:
                # Try unpacking
                return func(*input_val)
        return func(input_val)

    def _outputs_match(self, actual: Any, expected: Any) -> bool:
        """Check if actual output matches expected output.

        Handles numeric tolerance for floats.
        """
        if actual == expected:
            return True

        # Numeric tolerance
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(float(actual) - float(expected)) < 1e-9

        # String comparison
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.strip() == expected.strip()

        return False
