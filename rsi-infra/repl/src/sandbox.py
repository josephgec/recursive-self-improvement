"""Abstract base class and result types for the REPL sandbox system."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result of executing code in a sandbox."""

    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    variables: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None


_DEFAULT_ALLOWED_PACKAGES: list[str] = [
    "numpy", "pandas", "scipy", "sympy",
    "z3-solver", "scikit-learn", "matplotlib",
]


@dataclass
class REPLConfig:
    """Configuration for a sandboxed REPL instance."""

    timeout_seconds: int = 300
    max_memory_mb: int = 4096
    max_recursion_depth: int = 10
    network_access: bool = False
    gpu_enabled: bool = False
    allowed_packages: list[str] = field(default_factory=lambda: list(_DEFAULT_ALLOWED_PACKAGES))

    @classmethod
    def from_dict(cls, d: dict) -> REPLConfig:
        """Create a REPLConfig from a dictionary (e.g. parsed YAML)."""
        return cls(
            timeout_seconds=d.get("timeout_seconds", 300),
            max_memory_mb=d.get("max_memory_mb", 4096),
            max_recursion_depth=d.get("max_recursion_depth", 10),
            network_access=d.get("network_access", False),
            gpu_enabled=d.get("gpu_enabled", False),
            allowed_packages=d.get("allowed_packages", list(_DEFAULT_ALLOWED_PACKAGES)),
        )


class SandboxREPL(abc.ABC):
    """Abstract base class for sandboxed Python REPL implementations.

    Each backend (local, Docker, Modal) implements this interface to provide
    code execution with resource limits, security policies, and state
    management.
    """

    def __init__(self, config: REPLConfig | None = None, depth: int = 0) -> None:
        self._config = config or REPLConfig()
        self._depth = depth

    @property
    def depth(self) -> int:
        """Current nesting depth (0 = root REPL)."""
        return self._depth

    @property
    def config(self) -> REPLConfig:
        return self._config

    @abc.abstractmethod
    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code and return the result."""

    @abc.abstractmethod
    def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the REPL namespace."""

    @abc.abstractmethod
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the REPL namespace."""

    @abc.abstractmethod
    def list_variables(self) -> dict[str, Any]:
        """Return all user-defined variables in the namespace."""

    @abc.abstractmethod
    def spawn_child(self) -> SandboxREPL:
        """Create a child REPL with a deep-copied namespace.

        The child inherits variables but mutations do not affect the parent.
        Raises RecursionError if max_recursion_depth is exceeded.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all state and reset the REPL to a clean environment."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release all resources held by this REPL."""
