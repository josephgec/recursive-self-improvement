"""Result integrator: formats solver outputs for LLM consumption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class IntegrationContext:
    """Context for result integration."""
    original_problem: str
    reasoning_so_far: str = ""
    step_number: int = 0


class ResultIntegrator:
    """Integrates external solver results into natural language."""

    def integrate(self, solver_output: str, context: Optional[IntegrationContext] = None) -> str:
        """Format solver result for LLM consumption.

        Args:
            solver_output: Raw output from a solver tool.
            context: Optional context about the problem.

        Returns:
            Formatted string for the LLM to continue reasoning.
        """
        if not solver_output:
            return "The solver returned no output."

        # Handle satisfiability results
        if solver_output.strip() in ("sat", "unsat"):
            return self._integrate_sat_result(solver_output.strip(), context)

        # Handle equation solutions
        if "=" in solver_output and any(c.isalpha() for c in solver_output.split("=")[0]):
            return self._integrate_equation_result(solver_output, context)

        # Handle numeric results
        try:
            val = float(solver_output.strip())
            return self._integrate_numeric_result(val, context)
        except (ValueError, OverflowError):
            pass

        # Generic integration
        return f"The solver returned: {solver_output}"

    def _integrate_sat_result(self, result: str, context: Optional[IntegrationContext]) -> str:
        """Integrate satisfiability result."""
        if result == "sat":
            return "The formula is satisfiable, therefore the logical statement is consistent."
        else:
            return "The formula is unsatisfiable, therefore the logical statement contains a contradiction."

    def _integrate_equation_result(self, result: str, context: Optional[IntegrationContext]) -> str:
        """Integrate equation solution."""
        return f"Solving the equation gives {result}, therefore the answer is {result.split('=')[-1].strip()}."

    def _integrate_numeric_result(self, value: float, context: Optional[IntegrationContext]) -> str:
        """Integrate numeric result."""
        if value == int(value):
            return f"The computation yields {int(value)}, therefore the answer is {int(value)}."
        return f"The computation yields {value}, therefore the answer is {value}."
