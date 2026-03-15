"""Rollback mechanism for reverting failed modifications."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Generator

from src.core.state import AgentState, StateManager


class RollbackManager:
    """Manages checkpoints and rollback for state recovery."""

    def __init__(self, state_manager: StateManager) -> None:
        self.state_manager = state_manager
        self._checkpoints: list[tuple[str, Path]] = []

    def checkpoint(self, state: AgentState, label: str = "pre_modification") -> Path:
        """Save a checkpoint before modification."""
        path = self.state_manager.save_to_disk(state, label=label)
        self._checkpoints.append((state.state_id, path))
        return path

    def rollback(self, state_id: str | None = None) -> AgentState | None:
        """Rollback to a checkpoint."""
        if state_id:
            # Find specific checkpoint
            for sid, path in self._checkpoints:
                if sid == state_id:
                    return self.state_manager.load_from_disk(path)
        elif self._checkpoints:
            # Rollback to last checkpoint
            sid, path = self._checkpoints[-1]
            return self.state_manager.load_from_disk(path)
        return None

    def rollback_if_failed(
        self,
        state: AgentState,
        condition: bool,
    ) -> AgentState:
        """Rollback if condition indicates failure."""
        if condition and self._checkpoints:
            rolled_back = self.rollback()
            if rolled_back:
                return rolled_back
        return state

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)


class GuardedModification:
    """Context manager for guarded modifications with automatic rollback."""

    def __init__(
        self,
        rollback_manager: RollbackManager,
        state: AgentState,
    ) -> None:
        self.rollback_manager = rollback_manager
        self.state = state
        self._checkpoint_path: Path | None = None
        self._success = False

    def __enter__(self) -> GuardedModification:
        self._checkpoint_path = self.rollback_manager.checkpoint(self.state)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None or not self._success:
            # Rollback on exception or explicit failure
            self.rollback_manager.rollback()
        return False  # Don't suppress exceptions

    def mark_success(self) -> None:
        """Mark the modification as successful (prevents rollback on exit)."""
        self._success = True
