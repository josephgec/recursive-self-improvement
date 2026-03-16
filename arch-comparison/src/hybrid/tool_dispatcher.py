"""Tool dispatcher for routing to external solvers."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)


class ToolDispatcher:
    """Dispatches tool calls to appropriate solvers.

    Uses built-in mock implementations that simulate SymPy/Z3 behaviour
    without requiring actual subprocess calls.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {
            "sympy_solve": self._sympy_solve,
            "z3_check": self._z3_check,
            "simplify": self._simplify,
            "factor": self._factor,
            "expand": self._expand,
        }

    def dispatch(self, tool_name: str, tool_input: str) -> ToolResult:
        """Dispatch a tool call and return the result."""
        start = time.monotonic()
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
                execution_time=0.0,
            )
        try:
            output = self._tools[tool_name](tool_input)
            elapsed = time.monotonic() - start
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=str(output),
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=str(exc),
                execution_time=elapsed,
            )

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return tool definitions for LLM consumption."""
        return [
            {
                "name": "sympy_solve",
                "description": "Solve a mathematical equation symbolically",
                "parameters": {"expression": "string — equation to solve"},
            },
            {
                "name": "z3_check",
                "description": "Check satisfiability of a logical formula",
                "parameters": {"formula": "string — logical formula"},
            },
            {
                "name": "simplify",
                "description": "Simplify a mathematical expression",
                "parameters": {"expression": "string — expression to simplify"},
            },
            {
                "name": "factor",
                "description": "Factor a polynomial expression",
                "parameters": {"expression": "string — polynomial to factor"},
            },
            {
                "name": "expand",
                "description": "Expand a mathematical expression",
                "parameters": {"expression": "string — expression to expand"},
            },
        ]

    @property
    def available_tools(self) -> List[str]:
        return list(self._tools.keys())

    # ── Mock solver implementations ──

    @staticmethod
    def _sympy_solve(expression: str) -> str:
        """Mock SymPy solver: handles simple arithmetic and linear equations."""
        expression = expression.strip()
        # Handle "solve x + 3 = 7" style
        eq_match = re.search(r"(\w)\s*\+\s*(\d+)\s*=\s*(\d+)", expression)
        if eq_match:
            var = eq_match.group(1)
            a, b = int(eq_match.group(2)), int(eq_match.group(3))
            return f"{var} = {b - a}"

        # Handle "solve 2*x = 10" style
        mul_match = re.search(r"(\d+)\s*\*\s*(\w)\s*=\s*(\d+)", expression)
        if mul_match:
            coeff, var, val = int(mul_match.group(1)), mul_match.group(2), int(mul_match.group(3))
            if coeff != 0:
                return f"{var} = {val // coeff}"

        # Handle simple arithmetic "3 + 4"
        arith_match = re.match(r"^\s*(\d+)\s*([\+\-\*])\s*(\d+)\s*$", expression)
        if arith_match:
            a, op, b = int(arith_match.group(1)), arith_match.group(2), int(arith_match.group(3))
            if op == "+":
                return str(a + b)
            elif op == "-":
                return str(a - b)
            elif op == "*":
                return str(a * b)

        # Fallback
        return f"solved({expression})"

    @staticmethod
    def _z3_check(formula: str) -> str:
        """Mock Z3 checker: determines satisfiability heuristically."""
        formula_lower = formula.lower().strip()
        # Simple contradiction detection
        if "false" in formula_lower or "contradiction" in formula_lower:
            return "unsat"
        if "true" in formula_lower or "tautology" in formula_lower:
            return "sat"
        # Check for "x > 5 and x < 3" style contradictions
        if re.search(r">\s*(\d+).*<\s*(\d+)", formula):
            m = re.search(r">\s*(\d+).*<\s*(\d+)", formula)
            if m and int(m.group(1)) >= int(m.group(2)):
                return "unsat"
        return "sat"

    @staticmethod
    def _simplify(expression: str) -> str:
        """Mock simplify."""
        expression = expression.strip()
        arith_match = re.match(r"^\s*(\d+)\s*([\+\-\*])\s*(\d+)\s*$", expression)
        if arith_match:
            a, op, b = int(arith_match.group(1)), arith_match.group(2), int(arith_match.group(3))
            if op == "+":
                return str(a + b)
            elif op == "-":
                return str(a - b)
            elif op == "*":
                return str(a * b)
        return f"simplified({expression})"

    @staticmethod
    def _factor(expression: str) -> str:
        """Mock factor."""
        return f"factored({expression.strip()})"

    @staticmethod
    def _expand(expression: str) -> str:
        """Mock expand."""
        return f"expanded({expression.strip()})"
