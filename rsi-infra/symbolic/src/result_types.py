"""Structured result types for symbolic math execution and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SymPyResult:
    """Result of executing code in the SymPy runner."""

    success: bool = True
    expression: str | None = None
    numeric_value: float | None = None
    latex: str | None = None
    steps: list[str] = field(default_factory=list)
    error: str | None = None
    execution_time_ms: float = 0.0


@dataclass
class Z3Result:
    """Result of executing code in the Z3 runner."""

    satisfiable: bool | None = None  # True=SAT, False=UNSAT, None=unknown/error
    model: dict[str, Any] = field(default_factory=dict)
    proof: str | None = None
    error: str | None = None
    execution_time_ms: float = 0.0


@dataclass
class TestCase:
    """A single test case with inputs and expected output."""

    inputs: dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None


@dataclass
class VerificationSpec:
    """Specification describing how to verify a solution."""

    mode: str = "execution"  # "execution" | "symbolic" | "formal"
    expected_output: Any = None
    expected_expression: str | None = None
    constraints: list[str] = field(default_factory=list)
    test_cases: list[TestCase] = field(default_factory=list)
    tolerance: float = 1e-9


@dataclass
class VerificationResult:
    """Result of verifying a solution against a specification."""

    passed: bool = False
    mode: str = "execution"
    details: str = ""
    error_location: str | None = None
    suggestion: str | None = None
    execution_result: SymPyResult | Z3Result | None = None
