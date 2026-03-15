"""Tests for the Z3 SMT solver runner."""

from __future__ import annotations

import pytest

from symbolic.src.z3_runner import Z3Runner


@pytest.fixture
def runner() -> Z3Runner:
    return Z3Runner(timeout=30.0)


class TestExecute:
    def test_sat_simple(self, runner: Z3Runner) -> None:
        code = """\
x = Int('x')
y = Int('y')
s = Solver()
s.add(x + y == 10)
s.add(x > 0)
s.add(y > 0)
"""
        result = runner.execute(code)
        assert result.satisfiable is True
        assert result.error is None
        # Model should have x and y with valid values
        assert "x" in result.model
        assert "y" in result.model
        x_val = int(result.model["x"])
        y_val = int(result.model["y"])
        assert x_val + y_val == 10
        assert x_val > 0
        assert y_val > 0

    def test_unsat_simple(self, runner: Z3Runner) -> None:
        code = """\
x = Int('x')
s = Solver()
s.add(x > 0)
s.add(x < 0)
"""
        result = runner.execute(code)
        assert result.satisfiable is False
        assert result.error is None

    def test_execution_time_recorded(self, runner: Z3Runner) -> None:
        code = """\
x = Int('x')
s = Solver()
s.add(x == 42)
"""
        result = runner.execute(code)
        assert result.execution_time_ms > 0


class TestCheckImplication:
    def test_implies_true(self, runner: Z3Runner) -> None:
        # x > 5 implies x > 3
        assert runner.check_implication(["x > 5"], "x > 3")

    def test_implies_false(self, runner: Z3Runner) -> None:
        # x > 5 does NOT imply x > 10
        assert not runner.check_implication(["x > 5"], "x > 10")

    def test_multiple_premises(self, runner: Z3Runner) -> None:
        # x > 0 and x < 5 implies x < 10
        assert runner.check_implication(["x > 0", "x < 5"], "x < 10")

    def test_multiple_premises_false(self, runner: Z3Runner) -> None:
        # x > 0 and x < 5 does NOT imply x > 4
        assert not runner.check_implication(["x > 0", "x < 5"], "x > 4")


class TestVerifyProgramProperty:
    def test_valid_property(self, runner: Z3Runner) -> None:
        # Pre: x > 0, Prog: y = x + 1, Post: y > 1
        result = runner.verify_program_property(
            preconditions=["x > 0"],
            postconditions=["y > 1"],
            program_constraints=["y == x + 1"],
        )
        # UNSAT means postconditions always hold
        assert result.satisfiable is False

    def test_invalid_property(self, runner: Z3Runner) -> None:
        # Pre: x > 0, Prog: y = x + 1, Post: y > 10 (not always true)
        result = runner.verify_program_property(
            preconditions=["x > 0"],
            postconditions=["y > 10"],
            program_constraints=["y == x + 1"],
        )
        # SAT means there exists a counter-example
        assert result.satisfiable is True


class TestErrorHandling:
    def test_syntax_error(self, runner: Z3Runner) -> None:
        result = runner.execute("def f(:")
        assert result.satisfiable is None
        assert result.error is not None

    def test_runtime_error(self, runner: Z3Runner) -> None:
        result = runner.execute("1 / 0")
        assert result.satisfiable is None
        assert result.error is not None
