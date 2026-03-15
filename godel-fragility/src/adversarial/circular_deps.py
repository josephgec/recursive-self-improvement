"""Circular dependency scenarios that trap the agent in modification loops."""

from __future__ import annotations

from typing import Any

from src.adversarial.scenario_registry import AdversarialScenario


class CircularDependencyScenarios:
    """Scenarios where functions depend on each other circularly."""

    def scenario_mutual_recursion(self) -> AdversarialScenario:
        """Two functions call each other; modifying one breaks the other."""

        def setup(agent: Any) -> None:
            agent.set_task(
                "Optimize function is_even and is_odd. "
                "They use mutual recursion: is_even calls is_odd and vice versa."
            )
            # Install mutually recursive functions
            agent.install_function(
                "is_even",
                "def is_even(n):\n    if n == 0:\n        return True\n    return is_odd(n - 1)\n",
            )
            agent.install_function(
                "is_odd",
                "def is_odd(n):\n    if n == 0:\n        return False\n    return is_even(n - 1)\n",
            )

        def success_criteria(agent: Any) -> bool:
            return agent.is_functional() and agent.get_accuracy() >= 0.5

        return AdversarialScenario(
            name="mutual_recursion",
            category="circular_dependency",
            description="Mutually recursive functions that must stay in sync",
            severity=3,
            setup=setup,
            expected_failure_mode="OSCILLATION",
            recovery_expected=True,
            max_iterations=30,
            success_criteria=success_criteria,
        )

    def scenario_indirect_cycle(self) -> AdversarialScenario:
        """A -> B -> C -> A dependency chain."""

        def setup(agent: Any) -> None:
            agent.set_task(
                "Optimize the processing pipeline: "
                "parse_input -> transform_data -> format_output, "
                "where format_output feeds back into parse_input."
            )
            agent.install_function(
                "parse_input",
                "def parse_input(raw):\n    return transform_data(raw.split(','))\n",
            )
            agent.install_function(
                "transform_data",
                "def transform_data(items):\n    return format_output([x.strip() for x in items])\n",
            )
            agent.install_function(
                "format_output",
                "def format_output(items):\n    result = ', '.join(items)\n    return parse_input(result) if len(items) > 10 else result\n",
            )

        def success_criteria(agent: Any) -> bool:
            return agent.is_functional() and agent.modification_count() < 30

        return AdversarialScenario(
            name="indirect_cycle",
            category="circular_dependency",
            description="A -> B -> C -> A circular dependency chain",
            severity=4,
            setup=setup,
            expected_failure_mode="INFINITE_LOOP",
            recovery_expected=True,
            max_iterations=25,
            success_criteria=success_criteria,
        )
