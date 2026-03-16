"""Child REPL memory management for namespace inheritance."""

from typing import Any, Dict, Set


class ChildMemoryManager:
    """Manages memory inheritance between parent and child REPLs.

    Prepares child namespaces by copying parent variables while
    excluding internal state like CONTEXT.
    """

    # Variables that should not be inherited by child REPLs
    EXCLUDED_NAMES: Set[str] = {
        "CONTEXT",
        "__builtins__",
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__FINAL_RESULT__",
        "__FINAL_VAR_NAME__",
    }

    def __init__(self, exclude_names: Set[str] = None):
        self._exclude = exclude_names or self.EXCLUDED_NAMES

    def prepare_child_namespace(
        self,
        parent_namespace: Dict[str, Any],
        include_functions: bool = True,
    ) -> Dict[str, Any]:
        """Prepare a namespace for a child REPL.

        Copies parent variables while excluding internal state.

        Args:
            parent_namespace: The parent REPL's namespace.
            include_functions: Whether to include function definitions.

        Returns:
            New namespace dictionary for the child.
        """
        child_ns: Dict[str, Any] = {}

        for name, value in parent_namespace.items():
            # Skip excluded names
            if name in self._exclude:
                continue

            # Skip dunder names
            if name.startswith("__") and name.endswith("__"):
                continue

            # Optionally skip functions
            if not include_functions and callable(value):
                continue

            # Copy the value (shallow copy)
            child_ns[name] = value

        return child_ns

    def load_into_child(
        self,
        child_namespace: Dict[str, Any],
        parent_namespace: Dict[str, Any],
        variable_names: list = None,
    ) -> None:
        """Load specific variables from parent into child namespace.

        Args:
            child_namespace: The child REPL's namespace to modify.
            parent_namespace: The parent REPL's namespace.
            variable_names: Specific names to load, or None for all.
        """
        if variable_names is None:
            prepared = self.prepare_child_namespace(parent_namespace)
            child_namespace.update(prepared)
        else:
            for name in variable_names:
                if name in parent_namespace and name not in self._exclude:
                    child_namespace[name] = parent_namespace[name]
