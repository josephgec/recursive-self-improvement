"""Solution verification harness that dispatches to SymPy or Z3."""

from __future__ import annotations

from symbolic.src.result_types import (
    SymPyResult,
    TestCase,
    VerificationResult,
    VerificationSpec,
    Z3Result,
)
from symbolic.src.sympy_runner import SymPyRunner
from symbolic.src.z3_runner import Z3Runner


class SolutionVerifier:
    """Verify code solutions against a specification.

    Supports three verification modes:

    * **execution** -- run the code, compare output against expected value
      or execute test cases.
    * **symbolic** -- compare symbolic expressions via SymPy.
    * **formal** -- prove properties via Z3.

    Parameters
    ----------
    sympy_runner:
        Runner for SymPy-based execution/verification.
    z3_runner:
        Runner for Z3-based formal verification.
    """

    def __init__(
        self,
        sympy_runner: SymPyRunner | None = None,
        z3_runner: Z3Runner | None = None,
    ) -> None:
        self.sympy_runner = sympy_runner or SymPyRunner()
        self.z3_runner = z3_runner or Z3Runner()

    def verify(
        self,
        solution_code: str,
        specification: VerificationSpec,
    ) -> VerificationResult:
        """Dispatch to the appropriate verification mode."""
        mode = specification.mode

        if mode == "execution":
            if specification.test_cases:
                return self.verify_with_test_cases(
                    solution_code, specification.test_cases
                )
            return self.verify_math_solution(
                solution_code, specification.expected_output
            )
        elif mode == "symbolic":
            return self._verify_symbolic(solution_code, specification)
        elif mode == "formal":
            return self._verify_formal(solution_code, specification)
        else:
            return VerificationResult(
                passed=False,
                mode=mode,
                details=f"Unknown verification mode: {mode!r}",
            )

    def verify_math_solution(
        self,
        code: str,
        expected_answer: object,
    ) -> VerificationResult:
        """Execute *code* via SymPy and compare the result to *expected_answer*.

        The comparison first tries exact string matching, then symbolic
        equality, then numeric comparison.
        """
        result = self.sympy_runner.execute(code)

        if not result.success:
            return VerificationResult(
                passed=False,
                mode="execution",
                details=f"Execution failed: {result.error}",
                error_location="execution",
                suggestion="Fix the syntax or runtime error in the solution code.",
                execution_result=result,
            )

        expected_str = str(expected_answer)
        actual = result.expression

        # 1. Exact string match
        if actual == expected_str:
            return VerificationResult(
                passed=True,
                mode="execution",
                details=f"Result matches expected: {expected_str}",
                execution_result=result,
            )

        # 2. Symbolic equality
        if actual is not None:
            try:
                if self.sympy_runner.verify_equality(actual, expected_str):
                    return VerificationResult(
                        passed=True,
                        mode="execution",
                        details=(
                            f"Symbolically equivalent: {actual} == {expected_str}"
                        ),
                        execution_result=result,
                    )
            except Exception:
                pass

        # 3. Numeric comparison
        if result.numeric_value is not None:
            try:
                expected_num = float(expected_str)
                if abs(result.numeric_value - expected_num) < 1e-9:
                    return VerificationResult(
                        passed=True,
                        mode="execution",
                        details=(
                            f"Numerically equal: {result.numeric_value} "
                            f"~= {expected_num}"
                        ),
                        execution_result=result,
                    )
            except (ValueError, TypeError):
                pass

        return VerificationResult(
            passed=False,
            mode="execution",
            details=f"Expected {expected_str}, got {actual}",
            error_location="result",
            suggestion=(
                f"The solution produced '{actual}' but the expected answer is "
                f"'{expected_str}'. Check the mathematical reasoning."
            ),
            execution_result=result,
        )

    def verify_with_test_cases(
        self,
        code: str,
        test_cases: list[TestCase],
    ) -> VerificationResult:
        """Run *code* against each test case and report pass/fail.

        The code must define a function named ``solve`` that accepts keyword
        arguments matching the test case inputs.
        """
        failures: list[str] = []
        total = len(test_cases)

        for i, tc in enumerate(test_cases):
            # Build invocation code
            args_str = ", ".join(f"{k}={v!r}" for k, v in tc.inputs.items())
            invoke = f"{code}\nresult = solve({args_str})"

            result = self.sympy_runner.execute(invoke)

            if not result.success:
                failures.append(
                    f"Test {i + 1}: execution error -- {result.error}"
                )
                continue

            actual = result.expression
            expected_str = str(tc.expected_output)

            if actual != expected_str:
                # Try symbolic equality
                equal = False
                try:
                    equal = self.sympy_runner.verify_equality(actual or "", expected_str)
                except Exception:
                    pass

                if not equal:
                    failures.append(
                        f"Test {i + 1}: expected {expected_str}, got {actual}"
                    )

        if not failures:
            return VerificationResult(
                passed=True,
                mode="execution",
                details=f"All {total} test cases passed.",
            )

        return VerificationResult(
            passed=False,
            mode="execution",
            details=f"{len(failures)}/{total} test cases failed:\n"
            + "\n".join(failures),
            error_location="test_cases",
            suggestion="Review failing test cases and correct the solution logic.",
        )

    def generate_feedback(self, result: VerificationResult) -> str:
        """Generate natural-language feedback suitable for an LLM.

        Returns a concise summary describing what happened and, for
        failures, actionable suggestions.
        """
        lines: list[str] = []

        if result.passed:
            lines.append("PASSED: The solution is correct.")
            lines.append(f"Details: {result.details}")
        else:
            lines.append("FAILED: The solution did not pass verification.")
            lines.append(f"Details: {result.details}")
            if result.error_location:
                lines.append(f"Error location: {result.error_location}")
            if result.suggestion:
                lines.append(f"Suggestion: {result.suggestion}")

        if result.execution_result is not None:
            er = result.execution_result
            if isinstance(er, SymPyResult):
                if er.expression:
                    lines.append(f"Computed expression: {er.expression}")
                if er.error:
                    lines.append(f"Runtime error: {er.error}")
            elif isinstance(er, Z3Result):
                if er.satisfiable is not None:
                    sat_str = "satisfiable" if er.satisfiable else "unsatisfiable"
                    lines.append(f"SMT result: {sat_str}")
                if er.error:
                    lines.append(f"Runtime error: {er.error}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _verify_symbolic(
        self,
        code: str,
        spec: VerificationSpec,
    ) -> VerificationResult:
        """Symbolic verification: execute code and compare expression."""
        result = self.sympy_runner.execute(code)

        if not result.success:
            return VerificationResult(
                passed=False,
                mode="symbolic",
                details=f"Execution failed: {result.error}",
                error_location="execution",
                suggestion="Fix the code so it produces a valid symbolic expression.",
                execution_result=result,
            )

        if spec.expected_expression is None:
            return VerificationResult(
                passed=True,
                mode="symbolic",
                details="Code executed successfully (no expected expression to compare).",
                execution_result=result,
            )

        actual = result.expression or ""
        expected = spec.expected_expression

        try:
            equal = self.sympy_runner.verify_equality(actual, expected)
        except Exception:
            equal = False

        if equal:
            return VerificationResult(
                passed=True,
                mode="symbolic",
                details=f"Expressions are equal: {actual} == {expected}",
                execution_result=result,
            )

        return VerificationResult(
            passed=False,
            mode="symbolic",
            details=f"Expected expression {expected}, got {actual}",
            error_location="expression",
            suggestion=(
                f"The computed expression '{actual}' does not match "
                f"the expected '{expected}'."
            ),
            execution_result=result,
        )

    def _verify_formal(
        self,
        code: str,
        spec: VerificationSpec,
    ) -> VerificationResult:
        """Formal verification: run Z3 code and check constraints."""
        z3_result = self.z3_runner.execute(code)

        if z3_result.error:
            return VerificationResult(
                passed=False,
                mode="formal",
                details=f"Z3 execution failed: {z3_result.error}",
                error_location="execution",
                suggestion="Fix the Z3 code so it can be evaluated.",
                execution_result=z3_result,
            )

        # If constraints are specified, check each as an implication
        if spec.constraints:
            all_hold = True
            failed_constraints: list[str] = []
            for constraint in spec.constraints:
                holds = self.z3_runner.check_implication([], constraint)
                if not holds:
                    all_hold = False
                    failed_constraints.append(constraint)

            if not all_hold:
                return VerificationResult(
                    passed=False,
                    mode="formal",
                    details=(
                        "Failed constraints: "
                        + ", ".join(failed_constraints)
                    ),
                    error_location="constraints",
                    suggestion="Some formal constraints are not satisfied.",
                    execution_result=z3_result,
                )

        # Default: report the SAT/UNSAT result
        if z3_result.satisfiable is False:
            details = "Z3 proved UNSAT (property holds)."
        elif z3_result.satisfiable is True:
            details = f"Z3 found SAT (counter-example): {z3_result.model}"
        else:
            details = "Z3 returned unknown."

        return VerificationResult(
            passed=z3_result.satisfiable is not None,
            mode="formal",
            details=details,
            execution_result=z3_result,
        )
