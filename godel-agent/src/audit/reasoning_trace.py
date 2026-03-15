"""Reasoning trace capture for debugging and analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEntry:
    """A single entry in a reasoning trace."""

    step: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReasoningTraceCapture:
    """Captures reasoning traces for debugging and analysis."""

    def __init__(self) -> None:
        self._traces: list[list[TraceEntry]] = []
        self._current_trace: list[TraceEntry] = []

    def capture(self, step: str, content: str, **metadata: Any) -> None:
        """Capture a reasoning step."""
        entry = TraceEntry(step=step, content=content, metadata=metadata)
        self._current_trace.append(entry)

    def end_trace(self) -> list[TraceEntry]:
        """End the current trace and start a new one."""
        trace = self._current_trace
        if trace:
            self._traces.append(trace)
        self._current_trace = []
        return trace

    def get_trace(self, index: int = -1) -> list[TraceEntry]:
        """Get a specific trace by index."""
        if not self._traces:
            return self._current_trace
        try:
            return self._traces[index]
        except IndexError:
            return []

    def get_all_traces(self) -> list[list[TraceEntry]]:
        """Get all completed traces."""
        return list(self._traces)

    def format_trace(self, trace: list[TraceEntry] | None = None) -> str:
        """Format a trace for display."""
        if trace is None:
            trace = self._current_trace

        lines: list[str] = []
        for i, entry in enumerate(trace):
            lines.append(f"[{i+1}] {entry.step}")
            lines.append(f"    {entry.content[:200]}")
            if entry.metadata:
                for k, v in entry.metadata.items():
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)
