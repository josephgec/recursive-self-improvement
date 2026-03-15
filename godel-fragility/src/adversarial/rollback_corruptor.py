"""Scenarios that corrupt the agent's rollback/checkpoint mechanism."""

from __future__ import annotations

from typing import Any

from src.adversarial.scenario_registry import AdversarialScenario


class RollbackCorruptionScenarios:
    """Attack the agent's ability to roll back bad modifications."""

    def scenario_corrupt_checkpoint_format(self) -> AdversarialScenario:
        """Corrupt the serialized checkpoint so rollback loads garbage."""

        def setup(agent: Any) -> None:
            agent.set_task("Continue improving the solution.")
            # Corrupt the checkpoint data
            agent.corrupt_checkpoint(mode="format")

        def success_criteria(agent: Any) -> bool:
            # Agent should detect corruption and rebuild from scratch
            return agent.is_functional() and agent.get_accuracy() >= 0.3

        return AdversarialScenario(
            name="corrupt_checkpoint_format",
            category="rollback_corruption",
            description="Checkpoint file format is corrupted",
            severity=4,
            setup=setup,
            expected_failure_mode="ROLLBACK_FAILURE",
            recovery_expected=True,
            max_iterations=20,
            success_criteria=success_criteria,
        )

    def scenario_corrupt_baseline_score(self) -> AdversarialScenario:
        """Corrupt the stored baseline score so the agent thinks bad code is good."""

        def setup(agent: Any) -> None:
            agent.set_task("Continue improving the solution.")
            # Make the agent think its baseline accuracy is 0.0
            # so any modification looks like an improvement
            agent.corrupt_baseline(new_score=0.0)

        def success_criteria(agent: Any) -> bool:
            return agent.get_accuracy() >= 0.4

        return AdversarialScenario(
            name="corrupt_baseline_score",
            category="rollback_corruption",
            description="Baseline score corrupted to always accept changes",
            severity=5,
            setup=setup,
            expected_failure_mode="SILENT_DEGRADATION",
            recovery_expected=False,
            max_iterations=25,
            success_criteria=success_criteria,
        )

    def scenario_slow_rollback(self) -> AdversarialScenario:
        """Make rollback take so long the agent times out."""

        def setup(agent: Any) -> None:
            agent.set_task("Continue improving the solution.")
            agent.set_rollback_delay(seconds=5.0)

        def success_criteria(agent: Any) -> bool:
            return agent.is_functional()

        return AdversarialScenario(
            name="slow_rollback",
            category="rollback_corruption",
            description="Rollback mechanism is artificially slowed",
            severity=2,
            setup=setup,
            expected_failure_mode="STAGNATION",
            recovery_expected=True,
            max_iterations=15,
            success_criteria=success_criteria,
        )
