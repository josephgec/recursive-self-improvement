"""Tests for the solution verification harness."""

from __future__ import annotations

import pytest

from symbolic.src.result_types import (
    TestCase,
    VerificationResult,
    VerificationSpec,
)
from symbolic.src.sympy_runner import SymPyRunner
from symbolic.src.verifier import SolutionVerifier
from symbolic.src.z3_runner import Z3Runner


@pytest.fixture
def verifier() -> SolutionVerifier:
    return SolutionVerifier(
        sympy_runner=SymPyRunner(timeout=30.0),
        z3_runner=Z3Runner(timeout=30.0),
    )


class TestVerifyMathSolution:
    def test_correct_solution(self, verifier: SolutionVerifier) -> None:
        code = "result = solve(x**2 - 4, x)"
        result = verifier.verify_math_solution(code, "[-2, 2]")
        assert result.passed
        assert result.mode == "execution"

    def test_incorrect_solution(self, verifier: SolutionVerifier) -> None:
        code = "result = solve(x**2 - 4, x)"
        result = verifier.verify_math_solution(code, "[1, 2, 3]")
        assert not result.passed
        assert result.suggestion is not None

    def test_numeric_match(self, verifier: SolutionVerifier) -> None:
        code = "result = pi"
        result = verifier.verify_math_solution(code, "pi")
        assert result.passed

    def test_execution_error(self, verifier: SolutionVerifier) -> None:
        code = "result = 1 / 0"
        result = verifier.verify_math_solution(code, "42")
        assert not result.passed
        assert result.error_location == "execution"


class TestVerifyWithTestCases:
    def test_all_pass(self, verifier: SolutionVerifier) -> None:
        code = """\
def solve(a=0, b=0):
    return a + b
"""
        test_cases = [
            TestCase(inputs={"a": 1, "b": 2}, expected_output=3),
            TestCase(inputs={"a": 10, "b": 20}, expected_output=30),
        ]
        result = verifier.verify_with_test_cases(code, test_cases)
        assert result.passed
        assert "2" in result.details  # "All 2 test cases passed"

    def test_some_fail(self, verifier: SolutionVerifier) -> None:
        code = """\
def solve(a=0, b=0):
    return a * b
"""
        test_cases = [
            TestCase(inputs={"a": 2, "b": 3}, expected_output=6),  # passes
            TestCase(inputs={"a": 1, "b": 2}, expected_output=3),  # fails (1*2=2)
        ]
        result = verifier.verify_with_test_cases(code, test_cases)
        assert not result.passed
        assert "1/2" in result.details  # 1 out of 2 failed


class TestVerifyDispatch:
    def test_execution_mode(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(
            mode="execution",
            expected_output="[-2, 2]",
        )
        result = verifier.verify("result = solve(x**2 - 4, x)", spec)
        assert result.passed

    def test_symbolic_mode(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(
            mode="symbolic",
            expected_expression="x**2 + 2*x + 1",
        )
        result = verifier.verify("result = expand((x + 1)**2)", spec)
        assert result.passed
        assert result.mode == "symbolic"

    def test_unknown_mode(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(mode="quantum")
        result = verifier.verify("x = 1", spec)
        assert not result.passed
        assert "Unknown" in result.details


class TestGenerateFeedback:
    def test_pass_feedback(self, verifier: SolutionVerifier) -> None:
        result = VerificationResult(
            passed=True,
            mode="execution",
            details="Result matches expected: [-2, 2]",
        )
        feedback = verifier.generate_feedback(result)
        assert "PASSED" in feedback
        assert "correct" in feedback.lower()

    def test_fail_feedback(self, verifier: SolutionVerifier) -> None:
        result = VerificationResult(
            passed=False,
            mode="execution",
            details="Expected [1, 2, 3], got [-2, 2]",
            error_location="result",
            suggestion="Check the mathematical reasoning.",
        )
        feedback = verifier.generate_feedback(result)
        assert "FAILED" in feedback
        assert "suggestion" in feedback.lower() or "Suggestion" in feedback

    def test_feedback_includes_expression(self, verifier: SolutionVerifier) -> None:
        from symbolic.src.result_types import SymPyResult

        result = VerificationResult(
            passed=False,
            mode="execution",
            details="Expected 42, got 41",
            execution_result=SymPyResult(success=True, expression="41"),
        )
        feedback = verifier.generate_feedback(result)
        assert "41" in feedback

    def test_feedback_informative(self, verifier: SolutionVerifier) -> None:
        result = VerificationResult(
            passed=False,
            mode="execution",
            details="Expected 5, got 3",
            error_location="result",
            suggestion="The answer is wrong, review the calculation.",
        )
        feedback = verifier.generate_feedback(result)
        # Feedback should contain all key information
        assert "FAILED" in feedback
        assert "Expected 5, got 3" in feedback
        assert "review" in feedback.lower()

    def test_feedback_with_z3_result_sat(self, verifier: SolutionVerifier) -> None:
        """Feedback includes Z3 satisfiability info when present."""
        from symbolic.src.result_types import Z3Result

        z3r = Z3Result(satisfiable=True, model={"x": "5"})
        result = VerificationResult(
            passed=True,
            mode="formal",
            details="Z3 found SAT",
            execution_result=z3r,
        )
        feedback = verifier.generate_feedback(result)
        assert "satisfiable" in feedback

    def test_feedback_with_z3_result_unsat(self, verifier: SolutionVerifier) -> None:
        from symbolic.src.result_types import Z3Result

        z3r = Z3Result(satisfiable=False)
        result = VerificationResult(
            passed=True,
            mode="formal",
            details="Z3 proved UNSAT",
            execution_result=z3r,
        )
        feedback = verifier.generate_feedback(result)
        assert "unsatisfiable" in feedback

    def test_feedback_with_z3_error(self, verifier: SolutionVerifier) -> None:
        from symbolic.src.result_types import Z3Result

        z3r = Z3Result(satisfiable=None, error="Z3 crashed")
        result = VerificationResult(
            passed=False,
            mode="formal",
            details="Z3 failed",
            execution_result=z3r,
        )
        feedback = verifier.generate_feedback(result)
        assert "Z3 crashed" in feedback

    def test_feedback_with_sympy_error(self, verifier: SolutionVerifier) -> None:
        from symbolic.src.result_types import SymPyResult

        spr = SymPyResult(success=False, error="SyntaxError in code")
        result = VerificationResult(
            passed=False,
            mode="execution",
            details="Execution failed",
            execution_result=spr,
        )
        feedback = verifier.generate_feedback(result)
        assert "SyntaxError in code" in feedback

    def test_feedback_passed_no_execution_result(self, verifier: SolutionVerifier) -> None:
        result = VerificationResult(
            passed=True,
            mode="execution",
            details="All good",
        )
        feedback = verifier.generate_feedback(result)
        assert "PASSED" in feedback
        assert "correct" in feedback.lower()

    def test_feedback_failed_no_location_no_suggestion(self, verifier: SolutionVerifier) -> None:
        result = VerificationResult(
            passed=False,
            mode="execution",
            details="Something went wrong",
        )
        feedback = verifier.generate_feedback(result)
        assert "FAILED" in feedback
        assert "Something went wrong" in feedback


# ---------------------------------------------------------------------------
# Tests: execution mode dispatch with test_cases
# ---------------------------------------------------------------------------

class TestVerifyDispatchExecution:
    """Test the verify() dispatch for execution mode with test cases."""

    def test_execution_mode_with_test_cases(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(
            mode="execution",
            test_cases=[
                TestCase(inputs={"a": 1, "b": 2}, expected_output=3),
            ],
        )
        code = """\
def solve(a=0, b=0):
    return a + b
"""
        result = verifier.verify(code, spec)
        assert result.passed

    def test_execution_mode_without_test_cases(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(
            mode="execution",
            expected_output="[-2, 2]",
        )
        result = verifier.verify("result = solve(x**2 - 4, x)", spec)
        assert result.passed

    def test_execution_mode_test_case_failure(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(
            mode="execution",
            test_cases=[
                TestCase(inputs={"a": 1, "b": 2}, expected_output=99),
            ],
        )
        code = """\
def solve(a=0, b=0):
    return a + b
"""
        result = verifier.verify(code, spec)
        assert not result.passed


# ---------------------------------------------------------------------------
# Tests: formal mode dispatch
# ---------------------------------------------------------------------------

class TestFormalModeDispatch:
    """Test verify() dispatch to formal mode (Z3)."""

    def test_formal_mode_basic(self, verifier: SolutionVerifier) -> None:
        """Test formal verification with a simple Z3 problem."""
        spec = VerificationSpec(mode="formal")
        code = """
s = Solver()
x = Int('x')
s.add(x > 0, x < 5)
"""
        result = verifier.verify(code, spec)
        assert result.mode == "formal"
        # Should have a result (SAT or error)
        assert result.details is not None

    def test_formal_mode_with_constraints_passing(self, verifier: SolutionVerifier) -> None:
        """Formal mode with constraints that hold."""
        spec = VerificationSpec(
            mode="formal",
            constraints=["x > 3"],
        )
        code = """
s = Solver()
x = Int('x')
s.add(x > 5)
"""
        result = verifier.verify(code, spec)
        assert result.mode == "formal"

    def test_formal_mode_z3_error(self, verifier: SolutionVerifier) -> None:
        """Formal mode reports errors from Z3."""
        spec = VerificationSpec(mode="formal")
        # Invalid Z3 code that will cause an error
        code = "this is not valid z3 code at all!!!"
        result = verifier.verify(code, spec)
        assert result.mode == "formal"
        assert not result.passed


# ---------------------------------------------------------------------------
# Tests: symbolic mode dispatch
# ---------------------------------------------------------------------------

class TestSymbolicModeDispatch:
    """Test verify() dispatch to symbolic mode."""

    def test_symbolic_no_expected_expression(self, verifier: SolutionVerifier) -> None:
        """Symbolic mode with no expected expression just verifies execution."""
        spec = VerificationSpec(
            mode="symbolic",
            expected_expression=None,
        )
        result = verifier.verify("result = 2 + 2", spec)
        assert result.mode == "symbolic"
        assert result.passed

    def test_symbolic_matching_expression(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(
            mode="symbolic",
            expected_expression="x**2 + 2*x + 1",
        )
        result = verifier.verify("result = expand((x + 1)**2)", spec)
        assert result.passed

    def test_symbolic_mismatched_expression(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(
            mode="symbolic",
            expected_expression="x**3",
        )
        result = verifier.verify("result = x**2", spec)
        assert not result.passed
        assert result.mode == "symbolic"
        assert result.error_location == "expression"

    def test_symbolic_execution_failure(self, verifier: SolutionVerifier) -> None:
        spec = VerificationSpec(
            mode="symbolic",
            expected_expression="42",
        )
        result = verifier.verify("result = 1 / 0", spec)
        assert not result.passed
        assert result.mode == "symbolic"
        assert result.error_location == "execution"


# ---------------------------------------------------------------------------
# Tests: verify_math_solution edge cases
# ---------------------------------------------------------------------------

class TestVerifyMathSolutionEdgeCases:
    """Test edge cases in verify_math_solution."""

    def test_numeric_comparison_close(self, verifier: SolutionVerifier) -> None:
        """Numeric comparison within tolerance passes."""
        code = "result = 3.14159265358979"
        result = verifier.verify_math_solution(code, "3.14159265358979")
        assert result.passed

    def test_symbolic_equality_fallback(self, verifier: SolutionVerifier) -> None:
        """Symbolic equality is tried when string match fails."""
        # (x+1)^2 and x^2+2x+1 are symbolically equal but different strings
        code = "result = expand((x+1)**2)"
        result = verifier.verify_math_solution(code, "x**2 + 2*x + 1")
        assert result.passed


# ---------------------------------------------------------------------------
# Tests: verify_with_test_cases edge cases
# ---------------------------------------------------------------------------

class TestVerifyWithTestCasesEdgeCases:
    def test_execution_error_in_test_case(self, verifier: SolutionVerifier) -> None:
        """A test case where execution fails is reported as a failure."""
        code = """\
def solve(a=0):
    return 1 / 0
"""
        test_cases = [
            TestCase(inputs={"a": 1}, expected_output=1),
        ]
        result = verifier.verify_with_test_cases(code, test_cases)
        assert not result.passed
        assert "execution error" in result.details.lower() or "1/1" in result.details
