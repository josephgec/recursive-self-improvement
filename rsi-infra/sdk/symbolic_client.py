"""High-level symbolic client for rsi-infra.

Wraps the :class:`SymbolicSandbox` with convenient single-call methods.
"""

from __future__ import annotations

from typing import Any

from sdk.config import InfraConfig
from symbolic.src.result_types import (
    SymPyResult,
    VerificationResult,
    VerificationSpec,
    Z3Result,
)
from symbolic.src.sandbox import SymbolicConfig, SymbolicSandbox


class SymbolicClient:
    """Convenient facade for symbolic math and formal verification.

    Usage::

        client = SymbolicClient.from_config(config)
        result = client.solve("x**2 - 4", "x")
        assert result.success
    """

    def __init__(self, sandbox: SymbolicSandbox) -> None:
        self._sandbox = sandbox

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: InfraConfig) -> SymbolicClient:
        """Create a client from an :class:`InfraConfig`."""
        sym_config = config.symbolic_config
        backend = config.symbolic_backend
        sandbox = SymbolicSandbox(config=sym_config, backend=backend)
        return cls(sandbox)

    # ------------------------------------------------------------------
    # SymPy convenience methods
    # ------------------------------------------------------------------

    def solve(self, expression: str, variable: str) -> SymPyResult:
        """Solve *expression* == 0 for *variable* using SymPy.

        Returns a :class:`SymPyResult` with the solution as ``expression``.
        """
        code = (
            f"from sympy import symbols, solve\n"
            f"{variable} = symbols('{variable}')\n"
            f"result = solve({expression}, {variable})\n"
        )
        return self._sandbox.execute_sympy(code)

    def evaluate(self, code: str) -> SymPyResult:
        """Execute arbitrary SymPy code and return the result."""
        return self._sandbox.execute_sympy(code)

    # ------------------------------------------------------------------
    # Z3 convenience methods
    # ------------------------------------------------------------------

    def check_implication(
        self,
        premises: list[str],
        conclusion: str,
    ) -> bool:
        """Check whether *premises* logically imply *conclusion* via Z3.

        Returns ``True`` if the implication holds (UNSAT on negation).
        """
        return self._sandbox.z3_runner.check_implication(premises, conclusion)

    def check_sat(self, code: str) -> Z3Result:
        """Execute Z3 code and return the SAT/UNSAT result."""
        return self._sandbox.execute_z3(code)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_code(
        self,
        code: str,
        expected: Any,
        mode: str = "execution",
    ) -> VerificationResult:
        """Run *code* and verify the output matches *expected*.

        Parameters
        ----------
        code:
            Python/SymPy code to execute.
        expected:
            Expected result (compared via string, symbolic, and numeric).
        mode:
            Verification mode: ``"execution"``, ``"symbolic"``, or ``"formal"``.
        """
        spec = VerificationSpec(
            mode=mode,
            expected_output=expected,
            expected_expression=str(expected) if mode == "symbolic" else None,
        )
        return self._sandbox.verify(code, spec)

    def verify_with_spec(
        self,
        code: str,
        spec: VerificationSpec,
    ) -> VerificationResult:
        """Verify code against a full :class:`VerificationSpec`."""
        return self._sandbox.verify(code, spec)

    # ------------------------------------------------------------------
    # Direct access
    # ------------------------------------------------------------------

    @property
    def sandbox(self) -> SymbolicSandbox:
        """Access the underlying :class:`SymbolicSandbox`."""
        return self._sandbox
