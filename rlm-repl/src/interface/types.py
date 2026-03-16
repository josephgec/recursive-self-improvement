"""Data types for the RLM-REPL interface."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExecutionResult:
    """Result of executing code in the sandbox.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        error: Error message if execution failed, None otherwise.
        error_type: Type name of the error (e.g., 'ZeroDivisionError').
        execution_time_ms: Wall-clock execution time in milliseconds.
        memory_peak_mb: Peak memory usage in megabytes during execution.
        variables_changed: List of variable names that were created or modified.
        killed: Whether the execution was forcibly terminated.
        kill_reason: Reason the execution was killed (timeout, memory, etc.).
    """

    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    variables_changed: List[str] = field(default_factory=list)
    killed: bool = False
    kill_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether execution completed without errors."""
        return self.error is None and not self.killed
