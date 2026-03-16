"""CustomConstraint: user-defined constraint with a lambda/callable check function."""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.constraints.base import Constraint, ConstraintResult, CheckContext


class CustomConstraint(Constraint):
    """User-defined constraint backed by an arbitrary check callable.

    The callable receives ``(agent_state, context)`` and must return a
    ``ConstraintResult``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        threshold: float,
        check_fn: Callable[[Any, CheckContext], ConstraintResult],
        is_immutable: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            category=category,
            threshold=threshold,
            is_immutable=is_immutable,
        )
        self._check_fn = check_fn

    def check(self, agent_state: Any, context: CheckContext) -> ConstraintResult:
        return self._check_fn(agent_state, context)
