"""Component registry for tracking modifiable components."""

from __future__ import annotations

from typing import Any


class ComponentRegistry:
    """Registry for modifiable agent components.

    Tracks components that the self-modification loop can inspect and modify.
    """

    def __init__(self) -> None:
        self._components: dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        """Register a component by name."""
        self._components[name] = component

    def get(self, name: str) -> Any:
        """Retrieve a component by name. Raises KeyError if not found."""
        if name not in self._components:
            raise KeyError(f"Component '{name}' not registered")
        return self._components[name]

    def list_components(self) -> list[str]:
        """List all registered component names."""
        return list(self._components.keys())

    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components

    def replace(self, name: str, component: Any) -> Any:
        """Replace a component, returning the old one."""
        old = self._components.get(name)
        self._components[name] = component
        return old

    def __contains__(self, name: str) -> bool:
        return name in self._components

    def __len__(self) -> int:
        return len(self._components)
