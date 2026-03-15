"""Tests for the stress test runner."""

from __future__ import annotations

from typing import Any

import pytest

from src.adversarial.scenario_registry import AdversarialScenario, ScenarioRegistry
from src.harness.stress_runner import ScenarioResult, StressTestResults, StressTestRunner
from src.measurement.failure_classifier import FailureMode


@pytest.fixture
def simple_registry() -> ScenarioRegistry:
    """Create a registry with scenarios that complete quickly."""
    registry = ScenarioRegistry()

    def setup_pass(agent: Any) -> None:
        agent.set_task("Easy task")

    def success_pass(agent: Any) -> bool:
        return agent.get_accuracy() >= 0.5

    registry.register(
        AdversarialScenario(
            name="easy_scenario",
            category="test_cat",
            description="Should pass",
            severity=1,
            setup=setup_pass,
            expected_failure_mode="STAGNATION",
            recovery_expected=True,
            max_iterations=5,
            success_criteria=success_pass,
        )
    )

    def setup_fail(agent: Any) -> None:
        agent.set_task("Hard task")
        agent.corrupt_baseline(new_score=0.0)

    def success_fail(agent: Any) -> bool:
        return agent.get_accuracy() >= 0.99  # Unreachable

    registry.register(
        AdversarialScenario(
            name="hard_scenario",
            category="test_cat",
            description="Should fail",
            severity=4,
            setup=setup_fail,
            expected_failure_mode="SILENT_DEGRADATION",
            recovery_expected=False,
            max_iterations=5,
            success_criteria=success_fail,
        )
    )

    return registry


class TestStressTestRunner:
    def test_run_all(self, simple_registry: ScenarioRegistry) -> None:
        runner = StressTestRunner(
            registry=simple_registry,
            seed=42,
            timeout_seconds=10.0,
            repetitions=1,
        )
        results = runner.run_all()
        assert isinstance(results, StressTestResults)
        assert results.total_scenarios == 2
        assert results.duration_seconds > 0

    def test_run_single_scenario(self, simple_registry: ScenarioRegistry) -> None:
        runner = StressTestRunner(
            registry=simple_registry,
            seed=42,
            timeout_seconds=10.0,
        )
        result = runner.run_scenario("easy_scenario")
        assert isinstance(result, ScenarioResult)
        assert result.scenario_name == "easy_scenario"
        assert result.category == "test_cat"
        assert isinstance(result.failure_mode, FailureMode)

    def test_run_category(self, simple_registry: ScenarioRegistry) -> None:
        runner = StressTestRunner(
            registry=simple_registry,
            seed=42,
            timeout_seconds=10.0,
        )
        results = runner.run_category("test_cat")
        assert results.total_scenarios == 2

    def test_scenario_not_found(self, simple_registry: ScenarioRegistry) -> None:
        runner = StressTestRunner(
            registry=simple_registry,
            seed=42,
            timeout_seconds=10.0,
        )
        with pytest.raises(KeyError):
            runner.run_scenario("nonexistent")

    def test_scenario_result_has_accuracies(self, simple_registry: ScenarioRegistry) -> None:
        runner = StressTestRunner(
            registry=simple_registry,
            seed=42,
            timeout_seconds=10.0,
        )
        result = runner.run_scenario("easy_scenario")
        assert len(result.accuracies) > 0
        assert all(0 <= a <= 1 for a in result.accuracies)

    def test_scenario_result_has_complexities(self, simple_registry: ScenarioRegistry) -> None:
        runner = StressTestRunner(
            registry=simple_registry,
            seed=42,
            timeout_seconds=10.0,
        )
        result = runner.run_scenario("easy_scenario")
        assert len(result.complexities) > 0

    def test_repetitions(self, simple_registry: ScenarioRegistry) -> None:
        runner = StressTestRunner(
            registry=simple_registry,
            seed=42,
            timeout_seconds=10.0,
            repetitions=3,
        )
        results = runner.run_all()
        # 2 scenarios x 3 repetitions = 6
        assert results.total_scenarios == 6


class TestStressTestResults:
    def test_pass_rate(self) -> None:
        results = StressTestResults(
            total_scenarios=4,
            passed=3,
            failed=1,
        )
        assert results.pass_rate == 0.75

    def test_pass_rate_zero(self) -> None:
        results = StressTestResults(total_scenarios=0, passed=0, failed=0)
        assert results.pass_rate == 0.0

    def test_failure_mode_distribution(self) -> None:
        results = StressTestResults(
            results=[
                ScenarioResult(
                    scenario_name="a",
                    category="c",
                    severity=1,
                    success=False,
                    failure_mode=FailureMode.STAGNATION,
                ),
                ScenarioResult(
                    scenario_name="b",
                    category="c",
                    severity=2,
                    success=False,
                    failure_mode=FailureMode.STAGNATION,
                ),
                ScenarioResult(
                    scenario_name="c",
                    category="c",
                    severity=3,
                    success=False,
                    failure_mode=FailureMode.OSCILLATION,
                ),
            ]
        )
        dist = results.failure_mode_distribution
        assert dist["stagnation"] == 2
        assert dist["oscillation"] == 1

    def test_category_results(self) -> None:
        results = StressTestResults(
            results=[
                ScenarioResult("a", "cat1", 1, True, FailureMode.STAGNATION),
                ScenarioResult("b", "cat1", 2, False, FailureMode.STAGNATION),
                ScenarioResult("c", "cat2", 1, True, FailureMode.STAGNATION),
            ]
        )
        cats = results.category_results
        assert cats["cat1"]["passed"] == 1
        assert cats["cat1"]["failed"] == 1
        assert cats["cat2"]["passed"] == 1


class TestRecoveryTracking:
    def test_runner_tracks_recovery(self, simple_registry: ScenarioRegistry) -> None:
        runner = StressTestRunner(
            registry=simple_registry,
            seed=42,
            timeout_seconds=10.0,
        )
        runner.run_all()
        # At least the failing scenario should generate recovery events
        tracker = runner.recovery_tracker
        assert tracker is not None
