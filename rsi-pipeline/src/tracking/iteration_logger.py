"""Iteration logger: logs iteration results."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class IterationLogger:
    """Logs iteration results for tracking and analysis."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def log_iteration(self, result: Any) -> None:
        """Log an iteration result."""
        if hasattr(result, 'to_dict'):
            entry = result.to_dict()
        else:
            entry = {"raw": str(result)}
        self._history.append(entry)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get all logged iterations."""
        return list(self._history)

    def get_last(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get the last n logged iterations."""
        return list(self._history[-n:])

    def export(self, format: str = "json") -> str:
        """Export log history."""
        if format == "json":
            return json.dumps(self._history, indent=2)
        # Simple text format
        lines = []
        for entry in self._history:
            lines.append(str(entry))
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._history)

    def clear(self) -> None:
        """Clear all logged history."""
        self._history.clear()
