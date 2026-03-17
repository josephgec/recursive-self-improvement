"""Graduated constraint relaxation.

Allows constraints to be relaxed in controlled steps (max 3 steps,
2 percentage points each), with safety constraints never relaxed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Safety constraints that can NEVER be relaxed
SAFETY_CONSTRAINTS = frozenset({
    "no_harmful_output",
    "maintain_alignment",
    "preserve_safety_checks",
})


@dataclass
class RelaxationProposal:
    """A proposed constraint relaxation."""
    constraint_name: str
    current_value: float
    proposed_value: float
    step_number: int
    max_steps: int
    is_safety_constraint: bool
    approved: bool
    reason: str = ""

    @property
    def relaxation_amount(self) -> float:
        return self.proposed_value - self.current_value


class GraduatedRelaxation:
    """Graduated constraint relaxation system.

    Rules:
    - Maximum 3 relaxation steps per constraint
    - Each step relaxes by 2 percentage points (0.02)
    - Safety constraints can NEVER be relaxed
    - Relaxations can be reverted
    """

    def __init__(
        self,
        max_steps: int = 3,
        step_size_pp: int = 2,
        safety_constraints: Optional[set] = None,
    ):
        self.max_steps = max_steps
        self.step_size = step_size_pp / 100.0  # Convert pp to fraction
        self.safety_constraints = safety_constraints or SAFETY_CONSTRAINTS
        self._constraints: Dict[str, float] = {}
        self._original_values: Dict[str, float] = {}
        self._steps_taken: Dict[str, int] = {}
        self._history: List[RelaxationProposal] = []

    def set_constraint(self, name: str, value: float) -> None:
        """Set a constraint's initial value.

        Args:
            name: Constraint name.
            value: Initial constraint value (threshold).
        """
        self._constraints[name] = value
        if name not in self._original_values:
            self._original_values[name] = value
            self._steps_taken[name] = 0

    def propose_relaxation(
        self, constraint_name: str, reason: str = ""
    ) -> RelaxationProposal:
        """Propose relaxing a constraint by one step.

        Args:
            constraint_name: Name of the constraint to relax.
            reason: Why the relaxation is needed.

        Returns:
            RelaxationProposal with approval status.
        """
        if constraint_name not in self._constraints:
            raise KeyError(f"Unknown constraint: {constraint_name}")

        is_safety = constraint_name in self.safety_constraints
        current = self._constraints[constraint_name]
        steps = self._steps_taken.get(constraint_name, 0)

        if is_safety:
            return RelaxationProposal(
                constraint_name=constraint_name,
                current_value=current,
                proposed_value=current,
                step_number=steps,
                max_steps=self.max_steps,
                is_safety_constraint=True,
                approved=False,
                reason="Safety constraints cannot be relaxed",
            )

        if steps >= self.max_steps:
            return RelaxationProposal(
                constraint_name=constraint_name,
                current_value=current,
                proposed_value=current,
                step_number=steps,
                max_steps=self.max_steps,
                is_safety_constraint=False,
                approved=False,
                reason=f"Maximum relaxation steps ({self.max_steps}) reached",
            )

        proposed = current + self.step_size

        proposal = RelaxationProposal(
            constraint_name=constraint_name,
            current_value=current,
            proposed_value=proposed,
            step_number=steps + 1,
            max_steps=self.max_steps,
            is_safety_constraint=False,
            approved=True,
            reason=reason or "Graduated relaxation step",
        )
        self._history.append(proposal)
        return proposal

    def apply_relaxation(self, proposal: RelaxationProposal) -> bool:
        """Apply an approved relaxation proposal.

        Args:
            proposal: A RelaxationProposal to apply.

        Returns:
            True if applied, False if not approved.
        """
        if not proposal.approved:
            return False

        self._constraints[proposal.constraint_name] = proposal.proposed_value
        self._steps_taken[proposal.constraint_name] = proposal.step_number
        return True

    def revert(self, constraint_name: str) -> float:
        """Revert a constraint to its original value.

        Args:
            constraint_name: Name of the constraint to revert.

        Returns:
            The original value.

        Raises:
            KeyError: If constraint not found.
        """
        if constraint_name not in self._original_values:
            raise KeyError(f"Unknown constraint: {constraint_name}")

        original = self._original_values[constraint_name]
        self._constraints[constraint_name] = original
        self._steps_taken[constraint_name] = 0
        return original

    def can_relax_further(self, constraint_name: str) -> bool:
        """Check if a constraint can be relaxed further.

        Args:
            constraint_name: Name of the constraint.

        Returns:
            True if more relaxation steps are available.
        """
        if constraint_name in self.safety_constraints:
            return False
        if constraint_name not in self._constraints:
            return False
        steps = self._steps_taken.get(constraint_name, 0)
        return steps < self.max_steps

    def get_constraint_value(self, constraint_name: str) -> float:
        """Get current value of a constraint."""
        if constraint_name not in self._constraints:
            raise KeyError(f"Unknown constraint: {constraint_name}")
        return self._constraints[constraint_name]

    def get_all_constraints(self) -> Dict[str, float]:
        """Return all current constraint values."""
        return dict(self._constraints)

    def get_history(self) -> List[RelaxationProposal]:
        """Return relaxation history."""
        return list(self._history)
