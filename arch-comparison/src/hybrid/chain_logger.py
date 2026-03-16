"""Chain logger for recording and displaying reasoning chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LogEntry:
    """Single log entry in a reasoning chain."""
    step_number: int
    step_type: str  # "reasoning", "tool_call", "tool_result", "conclusion"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class ChainLogger:
    """Logs reasoning steps and formats them for display."""

    def __init__(self) -> None:
        self._chain: List[LogEntry] = []
        self._step_counter: int = 0

    def log_step(
        self,
        step_type: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[str] = None,
        tool_output: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> LogEntry:
        """Log a single reasoning step."""
        self._step_counter += 1
        entry = LogEntry(
            step_number=self._step_counter,
            step_type=step_type,
            content=content,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            metadata=metadata or {},
        )
        self._chain.append(entry)
        return entry

    def get_chain(self) -> List[LogEntry]:
        """Return the full reasoning chain."""
        return list(self._chain)

    def clear(self) -> None:
        """Clear the chain."""
        self._chain.clear()
        self._step_counter = 0

    def format_for_display(self) -> str:
        """Format the reasoning chain as a human-readable string."""
        if not self._chain:
            return "(empty chain)"

        lines: List[str] = []
        for entry in self._chain:
            prefix = f"[Step {entry.step_number}] ({entry.step_type})"
            lines.append(f"{prefix}: {entry.content}")
            if entry.tool_name:
                lines.append(f"  Tool: {entry.tool_name}")
            if entry.tool_input:
                lines.append(f"  Input: {entry.tool_input}")
            if entry.tool_output:
                lines.append(f"  Output: {entry.tool_output}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._chain)
