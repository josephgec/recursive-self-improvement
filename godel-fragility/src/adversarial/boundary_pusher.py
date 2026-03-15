"""Complexity escalation scenarios that push the agent past its limits."""

from __future__ import annotations

from typing import Any

from src.adversarial.scenario_registry import AdversarialScenario


class ComplexityEscalation:
    """Scenarios that ramp up code complexity to find the breaking point."""

    def scenario_forced_complexity_ramp(self) -> AdversarialScenario:
        """Steadily increase the complexity of the code the agent must manage."""

        def setup(agent: Any) -> None:
            agent.set_task(
                "Keep improving the solution while we steadily increase "
                "code complexity each iteration."
            )
            agent.set_complexity_schedule("linear", start=30, step=20)

        def success_criteria(agent: Any) -> bool:
            return agent.get_accuracy() >= 0.3

        return AdversarialScenario(
            name="forced_complexity_ramp",
            category="complexity_escalation",
            description="Linear complexity ramp to find ceiling",
            severity=3,
            setup=setup,
            expected_failure_mode="COMPLEXITY_EXPLOSION",
            recovery_expected=True,
            max_iterations=50,
            success_criteria=success_criteria,
        )

    def scenario_deep_nesting(self) -> AdversarialScenario:
        """Force the agent to work with deeply nested control structures."""

        def setup(agent: Any) -> None:
            agent.set_task(
                "Modify code that has very deeply nested if/for/while blocks."
            )
            agent.set_nesting_depth(15)

        def success_criteria(agent: Any) -> bool:
            return agent.get_accuracy() >= 0.3

        return AdversarialScenario(
            name="deep_nesting",
            category="complexity_escalation",
            description="Deeply nested control structures (15+ levels)",
            severity=3,
            setup=setup,
            expected_failure_mode="COMPLEXITY_EXPLOSION",
            recovery_expected=True,
            max_iterations=30,
            success_criteria=success_criteria,
        )

    def scenario_long_function_body(self) -> AdversarialScenario:
        """Force the agent to work with very long functions."""

        def setup(agent: Any) -> None:
            agent.set_task(
                "Optimize a function with hundreds of lines of logic."
            )
            agent.set_function_length(500)

        def success_criteria(agent: Any) -> bool:
            return agent.get_accuracy() >= 0.2

        return AdversarialScenario(
            name="long_function_body",
            category="complexity_escalation",
            description="Single function with 500+ lines",
            severity=2,
            setup=setup,
            expected_failure_mode="STAGNATION",
            recovery_expected=True,
            max_iterations=30,
            success_criteria=success_criteria,
        )

    def scenario_state_explosion(self) -> AdversarialScenario:
        """Code with exponentially many state combinations."""

        def setup(agent: Any) -> None:
            agent.set_task(
                "Modify code with many boolean flags creating exponential state space."
            )
            agent.set_state_variables(12)  # 2^12 = 4096 states

        def success_criteria(agent: Any) -> bool:
            return agent.get_accuracy() >= 0.2

        return AdversarialScenario(
            name="state_explosion",
            category="complexity_escalation",
            description="Exponential state space from boolean flags",
            severity=4,
            setup=setup,
            expected_failure_mode="COMPLEXITY_EXPLOSION",
            recovery_expected=False,
            max_iterations=20,
            success_criteria=success_criteria,
        )
