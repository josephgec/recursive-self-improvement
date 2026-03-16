"""Rollback bridge: checkpoint and rollback management."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from src.pipeline.state import PipelineState, AgentCodeSnapshot


class RollbackBridge:
    """Manages checkpoints and rollbacks for self-modification."""

    def __init__(self):
        self._checkpoints: List[AgentCodeSnapshot] = []

    def checkpoint(self, state: PipelineState) -> int:
        """Save current agent code as a checkpoint. Returns checkpoint depth."""
        snapshot = copy.deepcopy(state.agent_code)
        self._checkpoints.append(snapshot)
        return len(self._checkpoints)

    def rollback(self, state: PipelineState) -> bool:
        """Rollback to the most recent checkpoint. Returns True if successful."""
        if not self._checkpoints:
            return False
        previous = self._checkpoints.pop()
        state.agent_code = copy.deepcopy(previous)
        return True

    def get_checkpoint_depth(self) -> int:
        """Get the number of available checkpoints."""
        return len(self._checkpoints)

    def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()

    @property
    def checkpoints(self) -> List[AgentCodeSnapshot]:
        return list(self._checkpoints)
