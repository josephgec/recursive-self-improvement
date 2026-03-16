"""Pipeline lifecycle management: start, pause, resume, stop, checkpoint, restore."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from src.pipeline.state import PipelineState


class PipelineLifecycle:
    """Manages the lifecycle of the RSI pipeline."""

    def __init__(self, checkpoint_dir: str = "data/pipeline_state"):
        self._checkpoint_dir = checkpoint_dir
        self._checkpoints: List[str] = []
        self._state: Optional[PipelineState] = None

    @property
    def state(self) -> Optional[PipelineState]:
        return self._state

    def start(self, state: PipelineState) -> PipelineState:
        """Start the pipeline."""
        state.status = "running"
        state.metadata["started_at"] = time.time()
        self._state = state
        return state

    def pause(self, state: PipelineState) -> PipelineState:
        """Pause the pipeline."""
        state.status = "paused"
        state.metadata["paused_at"] = time.time()
        self._state = state
        return state

    def resume(self, state: PipelineState) -> PipelineState:
        """Resume a paused pipeline."""
        if state.status != "paused":
            raise ValueError(f"Cannot resume pipeline in state '{state.status}', must be 'paused'")
        state.status = "running"
        state.metadata["resumed_at"] = time.time()
        self._state = state
        return state

    def stop(self, state: PipelineState, reason: str = "") -> PipelineState:
        """Stop the pipeline."""
        state.status = "stopped"
        state.metadata["stopped_at"] = time.time()
        state.metadata["stop_reason"] = reason
        self._state = state
        return state

    def checkpoint(self, state: PipelineState) -> str:
        """Create a checkpoint and return its path."""
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        filename = f"checkpoint_{state.iteration}_{state.state_id}.json"
        path = os.path.join(self._checkpoint_dir, filename)
        state.save(path)
        self._checkpoints.append(path)
        return path

    def restore(self, path: str) -> PipelineState:
        """Restore state from a checkpoint."""
        state = PipelineState.load(path)
        self._state = state
        return state

    @property
    def checkpoints(self) -> List[str]:
        return list(self._checkpoints)
