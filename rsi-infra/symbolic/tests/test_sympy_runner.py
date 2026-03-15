"""Tests for the SymPy execution runner."""

from __future__ import annotations

import pytest

from symbolic.src.sympy_runner import SymPyRunner


@pytest.fixture
def runner() -> SymPyRunner:
    return SymPyRunner(timeout=30.0)


class TestExecute:
    def test_solve_quadratic(self, runner: SymPyRunner) -> None:
        result = runner.execute("result = solve(x**2 - 4, x)")
        assert result.success
        assert result.expression is not None
        # solve returns [-2, 2]; check both roots are present
        assert "-2" in result.expression
        assert "2" in result.expression

    def test_simplify_sqrt(self, runner: SymPyRunner) -> None:
        result = runner.execute("result = simplify(sqrt(8))")
        assert result.success
        assert result.expression is not None
        # sqrt(8) simplifies to 2*sqrt(2)
        assert "2" in result.expression
        assert "sqrt" in result.expression

    def test_capture_last_variable(self, runner: SymPyRunner) -> None:
        result = runner.execute("expr = sin(pi/6)")
        assert result.success
        # sin(pi/6) = 1/2
        assert result.expression is not None

    def test_numeric_value(self, runner: SymPyRunner) -> None:
        result = runner.execute("result = pi")
        assert result.success
        assert result.numeric_value is not None
        assert abs(result.numeric_value - 3.14159265) < 1e-6

    def test_latex_output(self, runner: SymPyRunner) -> None:
        result = runner.execute("result = Integral(x**2, x)")
        assert result.success
        assert result.latex is not None
        # LaTeX should contain integral notation
        assert "int" in result.latex or "\\int" in result.latex


class TestVerifyEquality:
    def test_trig_identity(self, runner: SymPyRunner) -> None:
        assert runner.verify_equality("sin(x)**2 + cos(x)**2", "1")

    def test_algebraic_identity(self, runner: SymPyRunner) -> None:
        assert runner.verify_equality("(x+1)**2", "x**2 + 2*x + 1")

    def test_not_equal(self, runner: SymPyRunner) -> None:
        assert not runner.verify_equality("x + 1", "x + 2")


class TestCheckNumeric:
    def test_numeric_match(self, runner: SymPyRunner) -> None:
        assert runner.check_numeric("sqrt(2)", 1.41421356237, tolerance=1e-6)

    def test_numeric_mismatch(self, runner: SymPyRunner) -> None:
        assert not runner.check_numeric("sqrt(2)", 2.0, tolerance=1e-6)


class TestErrorHandling:
    def test_syntax_error(self, runner: SymPyRunner) -> None:
        result = runner.execute("def f(:")
        assert not result.success
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error(self, runner: SymPyRunner) -> None:
        result = runner.execute("result = 1 / 0")
        assert not result.success
        assert result.error is not None

    def test_timeout(self) -> None:
        runner = SymPyRunner(timeout=3.0)
        result = runner.execute("import time; time.sleep(60)")
        assert not result.success
        assert result.error is not None
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_execution_time_recorded(self, runner: SymPyRunner) -> None:
        result = runner.execute("result = 1 + 1")
        assert result.execution_time_ms > 0
