"""Tests for the unified symbolic sandbox."""

from __future__ import annotations

import pytest

from symbolic.src.result_types import VerificationSpec
from symbolic.src.sandbox import SymbolicConfig, SymbolicSandbox


@pytest.fixture
def sandbox() -> SymbolicSandbox:
    return SymbolicSandbox(config=SymbolicConfig(timeout=30.0))


class TestSymPyThroughSandbox:
    def test_execute_sympy(self, sandbox: SymbolicSandbox) -> None:
        result = sandbox.execute_sympy("result = simplify(sqrt(8))")
        assert result.success
        assert result.expression is not None
        assert "sqrt" in result.expression

    def test_solve_equation(self, sandbox: SymbolicSandbox) -> None:
        result = sandbox.execute_sympy("result = solve(x**2 - 9, x)")
        assert result.success
        assert "-3" in result.expression
        assert "3" in result.expression


class TestZ3ThroughSandbox:
    def test_execute_z3(self, sandbox: SymbolicSandbox) -> None:
        code = """\
x = Int('x')
y = Int('y')
s = Solver()
s.add(x + y == 10)
s.add(x > 0)
s.add(y > 0)
"""
        result = sandbox.execute_z3(code)
        assert result.satisfiable is True
        assert "x" in result.model

    def test_unsat_z3(self, sandbox: SymbolicSandbox) -> None:
        code = """\
x = Int('x')
s = Solver()
s.add(x > 0)
s.add(x < 0)
"""
        result = sandbox.execute_z3(code)
        assert result.satisfiable is False


class TestVerifyThroughSandbox:
    def test_verify_execution_mode(self, sandbox: SymbolicSandbox) -> None:
        spec = VerificationSpec(
            mode="execution",
            expected_output="[-2, 2]",
        )
        result = sandbox.verify("result = solve(x**2 - 4, x)", spec)
        assert result.passed

    def test_verify_symbolic_mode(self, sandbox: SymbolicSandbox) -> None:
        spec = VerificationSpec(
            mode="symbolic",
            expected_expression="x**2 + 2*x + 1",
        )
        result = sandbox.verify("result = expand((x + 1)**2)", spec)
        assert result.passed

    def test_verify_failure(self, sandbox: SymbolicSandbox) -> None:
        spec = VerificationSpec(
            mode="execution",
            expected_output="42",
        )
        result = sandbox.verify("result = 7", spec)
        assert not result.passed


class TestSandboxConfig:
    def test_default_config(self) -> None:
        sandbox = SymbolicSandbox()
        assert sandbox.config.backend == "subprocess"
        assert sandbox.config.timeout == 30.0

    def test_custom_config(self) -> None:
        config = SymbolicConfig(timeout=10.0, max_memory_mb=1024)
        sandbox = SymbolicSandbox(config=config)
        assert sandbox.config.timeout == 10.0
        assert sandbox.config.max_memory_mb == 1024

    def test_from_dict(self) -> None:
        config = SymbolicConfig.from_dict({"timeout": 15.0, "backend": "subprocess"})
        assert config.timeout == 15.0
        assert config.backend == "subprocess"

    def test_backend_override(self) -> None:
        sandbox = SymbolicSandbox(backend="subprocess")
        assert sandbox.config.backend == "subprocess"

    def test_accessors(self) -> None:
        sandbox = SymbolicSandbox()
        assert sandbox.sympy_runner is not None
        assert sandbox.z3_runner is not None
        assert sandbox.verifier is not None
