"""Memory bridge: saves/loads state to/from RLM REPL."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.pipeline.state import PipelineState


class MemoryBridge:
    """Bridges pipeline state with an RLM REPL session's memory."""

    def __init__(self):
        self._repl_memory: Dict[str, Any] = {}

    def save_state_to_repl(self, state: PipelineState, key: str = "pipeline_state") -> None:
        """Save pipeline state into REPL memory."""
        self._repl_memory[key] = state.to_dict()

    def load_state_from_repl(self, key: str = "pipeline_state") -> Optional[PipelineState]:
        """Load pipeline state from REPL memory."""
        data = self._repl_memory.get(key)
        if data is None:
            return None
        return PipelineState.from_dict(data)

    def set(self, key: str, value: Any) -> None:
        """Set a value in REPL memory."""
        self._repl_memory[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from REPL memory."""
        return self._repl_memory.get(key, default)

    def keys(self) -> list:
        """List all keys in REPL memory."""
        return list(self._repl_memory.keys())

    def clear(self) -> None:
        """Clear all REPL memory."""
        self._repl_memory.clear()
