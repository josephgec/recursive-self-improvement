"""Unified symbolic sandbox — single entry point for SymPy, Z3, and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from symbolic.src.result_types import (
    SymPyResult,
    VerificationResult,
    VerificationSpec,
    Z3Result,
)
from symbolic.src.sympy_runner import SymPyRunner
from symbolic.src.verifier import SolutionVerifier
from symbolic.src.z3_runner import Z3Runner


@dataclass
class SymbolicConfig:
    """Configuration for the symbolic sandbox."""

    timeout: float = 30.0
    max_memory_mb: int = 2048
    backend: str = "subprocess"  # "subprocess" | "docker" (future)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SymbolicConfig:
        return cls(
            timeout=d.get("timeout", 30.0),
            max_memory_mb=d.get("max_memory_mb", 2048),
            backend=d.get("backend", "subprocess"),
        )


class SymbolicSandbox:
    """Unified facade for symbolic math execution and verification.

    Parameters
    ----------
    config:
        Configuration controlling timeouts, memory, and backend.
    backend:
        Override the backend from config (convenience parameter).
    """

    def __init__(
        self,
        config: SymbolicConfig | None = None,
        backend: str = "subprocess",
    ) -> None:
        self._config = config or SymbolicConfig(backend=backend)
        if backend != "subprocess":
            self._config.backend = backend

        self._sympy_runner = SymPyRunner(
            timeout=self._config.timeout,
            max_memory_mb=self._config.max_memory_mb,
        )
        self._z3_runner = Z3Runner(
            timeout=self._config.timeout,
            max_memory_mb=self._config.max_memory_mb,
        )
        self._verifier = SolutionVerifier(
            sympy_runner=self._sympy_runner,
            z3_runner=self._z3_runner,
        )

    @property
    def config(self) -> SymbolicConfig:
        return self._config

    @property
    def sympy_runner(self) -> SymPyRunner:
        return self._sympy_runner

    @property
    def z3_runner(self) -> Z3Runner:
        return self._z3_runner

    @property
    def verifier(self) -> SolutionVerifier:
        return self._verifier

    def execute_sympy(self, code: str) -> SymPyResult:
        """Execute SymPy code and return a structured result."""
        return self._sympy_runner.execute(code)

    def execute_z3(self, code: str) -> Z3Result:
        """Execute Z3 code and return a structured result."""
        return self._z3_runner.execute(code)

    def verify(
        self,
        code: str,
        spec: VerificationSpec,
    ) -> VerificationResult:
        """Verify a solution against a specification."""
        return self._verifier.verify(code, spec)
