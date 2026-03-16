"""Variable storage with diff tracking for sandboxed REPLs."""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class VariableDiff:
    """Difference between two variable store states.

    Attributes:
        added: Variable names that were added.
        modified: Variable names that were modified.
        removed: Variable names that were removed.
    """

    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    @property
    def changed(self) -> List[str]:
        """All variable names that changed."""
        return self.added + self.modified

    @property
    def has_changes(self) -> bool:
        """Whether any changes were detected."""
        return bool(self.added or self.modified or self.removed)


class VariableStore:
    """Manages variables for a sandboxed REPL.

    Provides CRUD operations, diff tracking, and size estimation.
    """

    # Names to exclude from user variable listing
    INTERNAL_NAMES: Set[str] = {
        "__builtins__", "__name__", "__doc__", "__package__",
        "__loader__", "__spec__", "__file__", "__cached__",
        "FINAL", "FINAL_VAR", "__FINAL_RESULT__", "__FINAL_VAR_NAME__",
        "CONTEXT",
    }

    def __init__(self):
        self._variables: Dict[str, Any] = {}

    def get(self, name: str) -> Any:
        """Get a variable value.

        Args:
            name: Variable name.

        Returns:
            The variable value.

        Raises:
            KeyError: If the variable does not exist.
        """
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' not found")
        return self._variables[name]

    def set(self, name: str, value: Any) -> None:
        """Set a variable value.

        Args:
            name: Variable name.
            value: Value to store.
        """
        self._variables[name] = value

    def delete(self, name: str) -> None:
        """Delete a variable.

        Args:
            name: Variable name.

        Raises:
            KeyError: If the variable does not exist.
        """
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' not found")
        del self._variables[name]

    def list_all(self) -> List[str]:
        """List all user-defined variable names.

        Returns:
            Sorted list of variable names, excluding internal names.
        """
        return sorted(
            name for name in self._variables
            if name not in self.INTERNAL_NAMES and not name.startswith("_")
        )

    def diff(self, previous: Dict[str, Any]) -> VariableDiff:
        """Compute the diff between current state and a previous state.

        Args:
            previous: Previous variable state to compare against.

        Returns:
            VariableDiff describing the changes.
        """
        current_keys = set(self._variables.keys()) - self.INTERNAL_NAMES
        previous_keys = set(previous.keys()) - self.INTERNAL_NAMES

        added = sorted(current_keys - previous_keys)
        removed = sorted(previous_keys - current_keys)
        modified = sorted(
            name for name in current_keys & previous_keys
            if not self._values_equal(self._variables[name], previous.get(name))
        )

        return VariableDiff(added=added, modified=modified, removed=removed)

    def _values_equal(self, a: Any, b: Any) -> bool:
        """Compare two values for equality, handling numpy arrays etc."""
        try:
            if type(a) != type(b):
                return False
            # Handle numpy arrays
            if hasattr(a, "__array__") and hasattr(a, "shape"):
                import numpy as np
                return np.array_equal(a, b)
            return a == b
        except Exception:
            return a is b

    def total_size_bytes(self) -> int:
        """Estimate total memory size of all stored variables.

        Returns:
            Estimated size in bytes.
        """
        total = 0
        for value in self._variables.values():
            total += self._estimate_size(value)
        return total

    def _estimate_size(self, obj: Any) -> int:
        """Estimate the memory size of an object."""
        try:
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in obj.items()
                )
            elif isinstance(obj, (list, tuple, set, frozenset)):
                size += sum(self._estimate_size(item) for item in obj)
            elif hasattr(obj, "nbytes"):
                size = obj.nbytes
            return size
        except Exception:
            return sys.getsizeof(None)

    def get_namespace(self) -> Dict[str, Any]:
        """Get the full namespace dictionary.

        Returns:
            Reference to the internal variable dictionary.
        """
        return self._variables

    def set_namespace(self, namespace: Dict[str, Any]) -> None:
        """Replace the entire namespace.

        Args:
            namespace: New namespace dictionary.
        """
        self._variables = dict(namespace)

    def snapshot(self) -> Dict[str, Any]:
        """Take a shallow copy of the current state.

        Returns:
            Copy of the variable dictionary.
        """
        return dict(self._variables)

    def clear(self) -> None:
        """Remove all variables."""
        self._variables.clear()
