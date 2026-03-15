"""Stress test runner that orchestrates adversarial scenarios."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.adversarial.scenario_registry import AdversarialScenario, ScenarioRegistry
from src.harness.controlled_env import ControlledEnvironment
from src.harness.snapshot_comparator import SnapshotComparator
from src.harness.timeout_guard import TimeoutGuard, TimeoutError
from src.measurement.failure_classifier import FailureClassifier, FailureMode
from src.measurement.health_monitor import AgentHealthMonitor
from src.measurement.recovery_tracker import RecoveryTracker


@dataclass
class ScenarioResult:
    """Result of running a single adversarial scenario."""

    scenario_name: str
    category: str
    severity: int
    success: bool
    failure_mode: FailureMode
    accuracies: List[float] = field(default_factory=list)
    complexities: List[int] = field(default_factory=list)
    modification_count: int = 0
    rollback_count: int = 0
    iterations_run: int = 0
    duration_seconds: float = 0.0
    timed_out: bool = False
    snapshot_comparison: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class StressTestResults:
    """Aggregate results from running all scenarios."""

    results: List[ScenarioResult] = field(default_factory=list)
    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    timed_out: int = 0
    duration_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.passed / self.total_scenarios

    @property
    def failure_mode_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for r in self.results:
            mode = r.failure_mode.value
            dist[mode] = dist.get(mode, 0) + 1
        return dist

    @property
    def category_results(self) -> Dict[str, Dict[str, int]]:
        cats: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            if r.category not in cats:
                cats[r.category] = {"passed": 0, "failed": 0, "total": 0}
            cats[r.category]["total"] += 1
            if r.success:
                cats[r.category]["passed"] += 1
            else:
                cats[r.category]["failed"] += 1
        return cats


class StressTestRunner:
    """Run adversarial stress tests against the agent."""

    def __init__(
        self,
        registry: ScenarioRegistry,
        seed: int = 42,
        timeout_seconds: float = 600.0,
        repetitions: int = 1,
    ) -> None:
        self._registry = registry
        self._seed = seed
        self._timeout = timeout_seconds
        self._repetitions = repetitions
        self._env = ControlledEnvironment(seed=seed)
        self._comparator = SnapshotComparator()
        self._classifier = FailureClassifier()
        self._recovery_tracker = RecoveryTracker()

    @property
    def recovery_tracker(self) -> RecoveryTracker:
        return self._recovery_tracker

    def run_all(self) -> StressTestResults:
        """Run all registered scenarios."""
        scenarios = self._registry.get_all()
        return self._run_scenarios(scenarios)

    def run_category(self, category: str) -> StressTestResults:
        """Run all scenarios in a category."""
        scenarios = self._registry.get_by_category(category)
        return self._run_scenarios(scenarios)

    def run_scenario(self, name: str) -> ScenarioResult:
        """Run a single scenario by name."""
        scenario = self._registry.get(name)
        results = []
        for _ in range(self._repetitions):
            result = self._execute_scenario(scenario)
            results.append(result)

        # Return the worst result across repetitions
        if not results:
            return ScenarioResult(
                scenario_name=name,
                category=scenario.category,
                severity=scenario.severity,
                success=False,
                failure_mode=FailureMode.STAGNATION,
            )

        # Pick the result with the lowest accuracy
        return min(results, key=lambda r: r.accuracies[-1] if r.accuracies else 0.0)

    def _run_scenarios(
        self, scenarios: List[AdversarialScenario]
    ) -> StressTestResults:
        """Run a list of scenarios and aggregate results."""
        start_time = time.monotonic()
        aggregate = StressTestResults()

        for scenario in scenarios:
            for _ in range(self._repetitions):
                result = self._execute_scenario(scenario)
                aggregate.results.append(result)
                aggregate.total_scenarios += 1
                if result.success:
                    aggregate.passed += 1
                elif result.timed_out:
                    aggregate.timed_out += 1
                    aggregate.failed += 1
                else:
                    aggregate.failed += 1

        aggregate.duration_seconds = time.monotonic() - start_time
        return aggregate

    def _execute_scenario(
        self, scenario: AdversarialScenario
    ) -> ScenarioResult:
        """Execute a single scenario in an isolated environment."""
        agent = self._env.create_fresh_agent()
        start_time = time.monotonic()

        # Take before snapshot
        before_snapshot = agent.get_snapshot()

        try:
            # Apply scenario setup
            self._env.apply_scenario(agent, scenario)

            # Run with timeout
            try:
                guard = TimeoutGuard(self._timeout)
                with guard:
                    iteration_results = self._env.run_iterations(agent, scenario)
            except TimeoutError:
                after_snapshot = agent.get_snapshot()
                comparison = self._comparator.compare(before_snapshot, after_snapshot)

                accuracies = [r["accuracy"] for r in (iteration_results if "iteration_results" in dir() else [])]  # noqa: F821
                return ScenarioResult(
                    scenario_name=scenario.name,
                    category=scenario.category,
                    severity=scenario.severity,
                    success=False,
                    failure_mode=FailureMode.INFINITE_LOOP,
                    accuracies=accuracies,
                    timed_out=True,
                    duration_seconds=time.monotonic() - start_time,
                    snapshot_comparison=comparison,
                )

            # Collect results
            accuracies = [r["accuracy"] for r in iteration_results]
            complexities = [r["complexity"] for r in iteration_results]

            # Take after snapshot
            after_snapshot = agent.get_snapshot()
            comparison = self._comparator.compare(before_snapshot, after_snapshot)

            # Check success
            success = False
            if scenario.success_criteria:
                try:
                    success = scenario.success_criteria(agent)
                except Exception:
                    success = False
            else:
                success = agent.is_functional() and (
                    not accuracies or accuracies[-1] >= 0.5
                )

            # Classify failure mode
            failure_mode = self._classifier.classify(
                accuracies=accuracies,
                complexities=complexities,
                modification_count=agent.modification_count(),
                rollback_count=agent._rollback_count,
                validation_rejections=agent._validation_rejections,
                agent_functional=agent.is_functional(),
                agent_can_modify=agent.can_modify(),
                max_iterations=scenario.max_iterations,
            )

            # Track recovery
            if not success:
                event = self._recovery_tracker.track_injection(
                    scenario_name=scenario.name,
                    fault_type=scenario.category,
                    iteration=0,
                    complexity=complexities[0] if complexities else 0,
                )
                # Check if it recovered at any point
                for i, acc in enumerate(accuracies):
                    detected = acc < before_snapshot.get("accuracy", 0.8) * 0.8
                    recovered = acc >= before_snapshot.get("accuracy", 0.8) * 0.9
                    self._recovery_tracker.update(
                        iteration=i,
                        accuracy=acc,
                        detected=detected,
                        detection_method="accuracy_drop" if detected else None,
                        recovered=recovered,
                        recovery_method="self_correction" if recovered else None,
                        recovery_quality=acc if recovered else 0.0,
                    )
                self._recovery_tracker.finalize_event()

            return ScenarioResult(
                scenario_name=scenario.name,
                category=scenario.category,
                severity=scenario.severity,
                success=success,
                failure_mode=failure_mode,
                accuracies=accuracies,
                complexities=complexities,
                modification_count=agent.modification_count(),
                rollback_count=agent._rollback_count,
                iterations_run=len(iteration_results),
                duration_seconds=time.monotonic() - start_time,
                snapshot_comparison=comparison,
            )

        except Exception as e:
            return ScenarioResult(
                scenario_name=scenario.name,
                category=scenario.category,
                severity=scenario.severity,
                success=False,
                failure_mode=FailureMode.STATE_CORRUPTION,
                duration_seconds=time.monotonic() - start_time,
                error=str(e),
            )
