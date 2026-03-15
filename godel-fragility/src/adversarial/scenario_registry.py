"""Adversarial scenario registry for stress-testing self-modifying agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AdversarialScenario:
    """A single adversarial scenario to run against the agent."""

    name: str
    category: str
    description: str
    severity: int  # 1 (mild) to 5 (catastrophic)
    setup: Callable[[Any], None]
    expected_failure_mode: str
    recovery_expected: bool = True
    max_iterations: int = 50
    success_criteria: Optional[Callable[[Any], bool]] = None

    def __post_init__(self) -> None:
        if not 1 <= self.severity <= 5:
            raise ValueError(f"Severity must be 1-5, got {self.severity}")


class ScenarioRegistry:
    """Central registry for all adversarial scenarios."""

    def __init__(self) -> None:
        self._scenarios: Dict[str, AdversarialScenario] = {}

    def register(self, scenario: AdversarialScenario) -> None:
        """Register a scenario by name."""
        self._scenarios[scenario.name] = scenario

    def get(self, name: str) -> AdversarialScenario:
        """Retrieve a scenario by name."""
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found. Available: {list(self._scenarios.keys())}")
        return self._scenarios[name]

    def get_by_category(self, category: str) -> List[AdversarialScenario]:
        """Get all scenarios in a category."""
        return [s for s in self._scenarios.values() if s.category == category]

    def get_by_severity(self, min_severity: int = 1, max_severity: int = 5) -> List[AdversarialScenario]:
        """Get scenarios within a severity range."""
        return [
            s for s in self._scenarios.values()
            if min_severity <= s.severity <= max_severity
        ]

    def get_all(self) -> List[AdversarialScenario]:
        """Get all registered scenarios."""
        return list(self._scenarios.values())

    def __len__(self) -> int:
        return len(self._scenarios)

    def __contains__(self, name: str) -> bool:
        return name in self._scenarios


# Global registry instance
_global_registry = ScenarioRegistry()


def get_global_registry() -> ScenarioRegistry:
    """Get the global scenario registry."""
    return _global_registry
