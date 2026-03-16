"""RollbackTrigger: triggers rollback on constraint failure."""

from __future__ import annotations

from typing import Any, Callable, List, Optional


class RollbackTrigger:
    """Triggers rollback when a constraint gate rejects a modification."""

    def __init__(self) -> None:
        self._rollback_manager: Optional[Any] = None
        self._history: List[dict] = []

    def set_rollback_manager(self, manager: Any) -> None:
        """Register a rollback manager.

        ``manager`` must expose ``rollback(reason: str) -> None``.
        """
        self._rollback_manager = manager

    def trigger(self, reason: str) -> dict:
        """Execute a rollback and record the event.

        Returns a dict describing the rollback event.
        """
        event = {
            "action": "rollback",
            "reason": reason,
            "rollback_executed": False,
        }

        if self._rollback_manager is not None:
            self._rollback_manager.rollback(reason)
            event["rollback_executed"] = True

        self._history.append(event)
        return event

    @property
    def history(self) -> List[dict]:
        return list(self._history)
