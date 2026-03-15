"""End-to-end integration tests.

Run 3 scenarios, verify recovery tracking, failure classification,
and fragility scoring all work together.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.adversarial.scenario_registry import AdversarialScenario, ScenarioRegistry
from src.analysis.failure_landscape import FailureLandscapeAnalyzer
from src.analysis.fragility_score import FragilityScorer
from src.harness.stress_runner import StressTestRunner
from src.measurement.failure_classifier import FailureClassifier, FailureMode
from src.measurement.recovery_tracker import RecoveryTracker


@pytest.fixture
def integration_registry() -> ScenarioRegistry:
    """Create a registry with 3 diverse scenarios for integration testing."""
    registry = ScenarioRegistry()

    # Scenario 1: Easy, should pass
    def setup_easy(agent: Any) -> None:
        agent.set_task("Easy task")

    def success_easy(agent: Any) -> bool:
        return agent.get_accuracy() >= 0.5

    registry.register(
        AdversarialScenario(
            name="easy_pass",
            category="basic",
            description="Easy scenario that should pass",
            severity=1,
            setup=setup_easy,
            expected_failure_mode="STAGNATION",
            max_iterations=5,
            success_criteria=success_easy,
        )
    )

    # Scenario 2: Medium, corrupts baseline
    def setup_medium(agent: Any) -> None:
        agent.set_task("Medium difficulty")
        agent.corrupt_baseline(new_score=0.1)

    def success_medium(agent: Any) -> bool:
        return agent.get_accuracy() >= 0.7

    registry.register(
        AdversarialScenario(
            name="medium_corrupt",
            category="corruption",
            description="Corrupts baseline score",
            severity=3,
            setup=setup_medium,
            expected_failure_mode="SILENT_DEGRADATION",
            max_iterations=10,
            success_criteria=success_medium,
        )
    )

    # Scenario 3: Hard, inverted scoring
    def setup_hard(agent: Any) -> None:
        agent.set_task("Hard task")
        agent.set_scoring_mode("inverted")

    def success_hard(agent: Any) -> bool:
        return agent.get_true_accuracy() >= 0.8

    registry.register(
        AdversarialScenario(
            name="hard_inverted",
            category="adversarial",
            description="Inverted performance signal",
            severity=5,
            setup=setup_hard,
            expected_failure_mode="SILENT_DEGRADATION",
            max_iterations=10,
            success_criteria=success_hard,
        )
    )

    return registry


class TestEndToEnd:
    def test_full_pipeline(self, integration_registry: ScenarioRegistry) -> None:
        """Run all 3 scenarios and verify the full pipeline."""
        runner = StressTestRunner(
            registry=integration_registry,
            seed=42,
            timeout_seconds=30.0,
            repetitions=1,
        )

        # Run all scenarios
        results = runner.run_all()

        # Basic checks
        assert results.total_scenarios == 3
        assert results.passed + results.failed == 3
        assert results.duration_seconds > 0

        # Every result has required fields
        for r in results.results:
            assert r.scenario_name
            assert r.category
            assert isinstance(r.failure_mode, FailureMode)
            assert r.iterations_run > 0
            assert len(r.accuracies) > 0

    def test_recovery_tracking_integration(
        self, integration_registry: ScenarioRegistry
    ) -> None:
        """Verify recovery tracker captures events from failed scenarios."""
        runner = StressTestRunner(
            registry=integration_registry,
            seed=42,
            timeout_seconds=30.0,
        )
        results = runner.run_all()
        tracker = runner.recovery_tracker

        # Should have events for failed scenarios
        failed_count = sum(1 for r in results.results if not r.success)
        if failed_count > 0:
            assert len(tracker.events) > 0

        # Recovery and detection rates should be valid
        assert 0.0 <= tracker.get_recovery_rate() <= 1.0
        assert 0.0 <= tracker.get_detection_rate() <= 1.0

    def test_failure_classification_integration(
        self, integration_registry: ScenarioRegistry
    ) -> None:
        """Verify failure classifier produces valid modes for all results."""
        runner = StressTestRunner(
            registry=integration_registry,
            seed=42,
            timeout_seconds=30.0,
        )
        results = runner.run_all()

        classifier = FailureClassifier()
        for r in results.results:
            # Verify the classified mode is valid
            assert r.failure_mode in FailureMode
            # Verify severity is in range
            sev = classifier.get_severity(r.failure_mode)
            assert 1 <= sev <= 5

    def test_fragility_score_integration(
        self, integration_registry: ScenarioRegistry
    ) -> None:
        """Verify fragility score computation from real results."""
        runner = StressTestRunner(
            registry=integration_registry,
            seed=42,
            timeout_seconds=30.0,
        )
        results = runner.run_all()

        scorer = FragilityScorer()
        report = scorer.compute(
            results,
            recovery_tracker=runner.recovery_tracker,
        )

        # Score should be between 0 and 1
        assert 0.0 <= report.overall_score <= 1.0

        # All components should be present
        assert "recovery_rate" in report.components
        assert "ceiling_ratio" in report.components
        assert "catastrophic_rate" in report.components
        assert "detection_rate" in report.components

        # Each component should be between 0 and 1
        for comp_name, comp_val in report.components.items():
            assert 0.0 <= comp_val <= 1.0, f"{comp_name} = {comp_val} out of range"

        # Interpretation and grade should be set
        assert report.interpretation
        assert report.grade
        assert report.recommendations

    def test_failure_landscape_integration(
        self, integration_registry: ScenarioRegistry
    ) -> None:
        """Verify failure landscape analysis from real results."""
        runner = StressTestRunner(
            registry=integration_registry,
            seed=42,
            timeout_seconds=30.0,
        )
        results = runner.run_all()

        analyzer = FailureLandscapeAnalyzer()
        landscape = analyzer.compute_landscape(results)

        assert landscape.total_scenarios == 3
        # Should have failure mode counts
        assert isinstance(landscape.failure_mode_counts, dict)
        # Should have category failure rates
        assert isinstance(landscape.category_failure_rates, dict)

    def test_scenario_isolation(
        self, integration_registry: ScenarioRegistry
    ) -> None:
        """Verify scenarios don't interfere with each other."""
        runner = StressTestRunner(
            registry=integration_registry,
            seed=42,
            timeout_seconds=30.0,
        )

        # Run easy scenario alone
        result_alone = runner.run_scenario("easy_pass")

        # Run all and compare easy result
        runner2 = StressTestRunner(
            registry=integration_registry,
            seed=42,
            timeout_seconds=30.0,
        )
        all_results = runner2.run_all()
        easy_result = next(
            r for r in all_results.results if r.scenario_name == "easy_pass"
        )

        # Results should be consistent (same seed, same setup)
        assert result_alone.success == easy_result.success
        assert result_alone.failure_mode == easy_result.failure_mode


class TestMockAgentBehavior:
    """Test that the MockAgent behaves correctly for integration."""

    def test_mock_agent_basic(self) -> None:
        from src.harness.controlled_env import MockAgent

        agent = MockAgent(seed=42)
        assert agent.is_functional()
        assert agent.can_modify()
        assert agent.get_accuracy() >= 0.0
        assert agent.get_complexity() > 0

    def test_mock_agent_modification(self) -> None:
        from src.harness.controlled_env import MockAgent

        agent = MockAgent(seed=42)
        original_code = agent.get_function_source("solve")
        new_code = "def solve(x):\n    return x + 1\n"
        success = agent.modify_function("solve", new_code)
        assert success
        assert agent.get_function_source("solve") == new_code
        assert agent.modification_count() == 1

    def test_mock_agent_rollback(self) -> None:
        from src.harness.controlled_env import MockAgent

        agent = MockAgent(seed=42)
        agent.modify_function("solve", "def solve(x):\n    return x + 1\n")
        original_acc = agent.get_accuracy()
        agent.rollback()
        # Should have rolled back

    def test_mock_agent_iteration(self) -> None:
        from src.harness.controlled_env import MockAgent

        agent = MockAgent(seed=42)
        accuracies = [agent.run_iteration() for _ in range(10)]
        assert len(accuracies) == 10
        assert all(0 <= a <= 1 for a in accuracies)

    def test_mock_agent_syntax_rejection(self) -> None:
        from src.harness.controlled_env import MockAgent

        agent = MockAgent(seed=42)
        success = agent.modify_function("solve", "def broken(")
        assert not success  # Should reject syntax errors
