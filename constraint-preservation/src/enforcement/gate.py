"""ConstraintGate: hard gate that blocks modifications on constraint failure.

There is NO override mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from src.checker.runner import ConstraintRunner
from src.checker.verdict import SuiteVerdict
from src.constraints.base import CheckContext


@dataclass(frozen=True)
class GateDecision:
    """Immutable record of a gate decision."""

    allowed: bool
    verdict: SuiteVerdict
    context: CheckContext
    reason: str = ""


class ConstraintGate:
    """Hard binary gate -- no override allowed.

    Call ``check_and_gate`` before any pipeline modification.  If it
    returns ``allowed=False`` the modification MUST NOT proceed.
    """

    def __init__(self, runner: ConstraintRunner) -> None:
        self._runner = runner

    def check_and_gate(
        self,
        agent_state: Any,
        context: Optional[CheckContext] = None,
    ) -> GateDecision:
        """Run all constraints and return a GateDecision."""
        if context is None:
            context = CheckContext()

        verdict = self._runner.run(agent_state, context)

        if verdict.passed:
            return GateDecision(
                allowed=True,
                verdict=verdict,
                context=context,
                reason="All constraints satisfied.",
            )

        violation_names = list(verdict.violations.keys())
        return GateDecision(
            allowed=False,
            verdict=verdict,
            context=context,
            reason=f"Constraint violations: {', '.join(violation_names)}",
        )

    def wrap_modification(
        self,
        modification_fn: Callable[..., Any],
        agent_state: Any,
        context: Optional[CheckContext] = None,
        *args: Any,
        **kwargs: Any,
    ) -> GateDecision:
        """Gate a modification: only execute ``modification_fn`` if all
        constraints pass.  Returns the GateDecision (with the fn result
        stored in ``context.metadata['modification_result']`` on success).
        """
        if context is None:
            context = CheckContext()

        decision = self.check_and_gate(agent_state, context)

        if decision.allowed:
            result = modification_fn(*args, **kwargs)
            # Store the result in a new context since GateDecision is frozen
            enriched_context = CheckContext(
                modification_type=context.modification_type,
                modification_description=context.modification_description,
                timestamp=context.timestamp,
                metadata={**context.metadata, "modification_result": result},
            )
            return GateDecision(
                allowed=True,
                verdict=decision.verdict,
                context=enriched_context,
                reason=decision.reason,
            )

        return decision
