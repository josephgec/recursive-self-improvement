"""Abstract base class for sandboxed REPL implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.interface.types import ExecutionResult


class SandboxREPL(ABC):
    """Abstract base class defining the sandbox REPL interface.

    All REPL backends must implement this interface to provide
    a consistent API for code execution within the RLM architecture.
    """

    @abstractmethod
    def execute(self, code: str) -> ExecutionResult:
        """Execute code in the sandbox and return the result.

        Args:
            code: Python code string to execute.

        Returns:
            ExecutionResult with stdout, stderr, timing, and variable changes.
        """
        ...

    @abstractmethod
    def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the sandbox namespace.

        Args:
            name: Variable name.

        Returns:
            The variable value.

        Raises:
            KeyError: If the variable does not exist.
        """
        ...

    @abstractmethod
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox namespace.

        Args:
            name: Variable name.
            value: Value to set.
        """
        ...

    @abstractmethod
    def list_variables(self) -> List[str]:
        """List all user-defined variable names in the sandbox.

        Returns:
            List of variable names.
        """
        ...

    @abstractmethod
    def spawn_child(self) -> "SandboxREPL":
        """Spawn a child REPL that inherits variables from this REPL.

        Returns:
            A new SandboxREPL instance.

        Raises:
            RecursionDepthError: If maximum spawn depth is exceeded.
        """
        ...

    @abstractmethod
    def snapshot(self) -> str:
        """Take a snapshot of the current REPL state.

        Returns:
            Snapshot identifier string.
        """
        ...

    @abstractmethod
    def restore(self, snapshot_id: str) -> None:
        """Restore the REPL to a previous snapshot.

        Args:
            snapshot_id: Identifier returned by snapshot().

        Raises:
            KeyError: If the snapshot does not exist.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the REPL to a clean state, clearing all variables."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the REPL, releasing all resources."""
        ...

    @abstractmethod
    def is_alive(self) -> bool:
        """Check whether the REPL is still running.

        Returns:
            True if the REPL is operational.
        """
        ...
