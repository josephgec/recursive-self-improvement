"""Tests for all adversarial scenario modules."""

from __future__ import annotations

import pytest

from src.adversarial.adversarial_tasks import AdversarialTaskScenarios
from src.adversarial.boundary_pusher import ComplexityEscalation
from src.adversarial.circular_deps import CircularDependencyScenarios
from src.adversarial.rollback_corruptor import RollbackCorruptionScenarios
from src.adversarial.scenario_registry import AdversarialScenario
from src.adversarial.self_reference import SelfReferenceAttacks


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _validate_scenario(scenario: AdversarialScenario) -> None:
    """Assert common invariants on an AdversarialScenario."""
    assert isinstance(scenario.name, str) and scenario.name
    assert isinstance(scenario.category, str) and scenario.category
    assert isinstance(scenario.description, str) and scenario.description
    assert 1 <= scenario.severity <= 5
    assert callable(scenario.setup)
    assert isinstance(scenario.expected_failure_mode, str)
    assert isinstance(scenario.recovery_expected, bool)
    assert scenario.max_iterations > 0
    # success_criteria may be None or callable
    assert scenario.success_criteria is None or callable(scenario.success_criteria)


# ------------------------------------------------------------------ #
# AdversarialTaskScenarios
# ------------------------------------------------------------------ #


class TestAdversarialTaskScenarios:
    def setup_method(self) -> None:
        self.cls = AdversarialTaskScenarios()

    def test_misleading_performance_signal(self) -> None:
        s = self.cls.scenario_misleading_performance_signal()
        _validate_scenario(s)
        assert s.name == "misleading_performance_signal"
        assert s.category == "adversarial_task"
        assert s.severity == 4
        assert s.recovery_expected is False

    def test_distribution_shift(self) -> None:
        s = self.cls.scenario_distribution_shift()
        _validate_scenario(s)
        assert s.name == "distribution_shift"
        assert s.category == "adversarial_task"
        assert s.severity == 3
        assert s.recovery_expected is True

    def test_impossible_tasks(self) -> None:
        s = self.cls.scenario_impossible_tasks()
        _validate_scenario(s)
        assert s.name == "impossible_tasks"
        assert s.category == "adversarial_task"
        assert s.severity == 3

    def test_gradual_poisoning(self) -> None:
        s = self.cls.scenario_gradual_poisoning()
        _validate_scenario(s)
        assert s.name == "gradual_poisoning"
        assert s.category == "adversarial_task"
        assert s.severity == 5
        assert s.recovery_expected is False

    def test_setup_callables_run(self, mock_agent) -> None:
        """Ensure setup callables run without error against a MockAgent."""
        for method_name in [
            "scenario_misleading_performance_signal",
            "scenario_distribution_shift",
            "scenario_impossible_tasks",
            "scenario_gradual_poisoning",
        ]:
            s = getattr(self.cls, method_name)()
            s.setup(mock_agent)  # should not raise


# ------------------------------------------------------------------ #
# ComplexityEscalation (boundary_pusher)
# ------------------------------------------------------------------ #


class TestComplexityEscalation:
    def setup_method(self) -> None:
        self.cls = ComplexityEscalation()

    def test_forced_complexity_ramp(self) -> None:
        s = self.cls.scenario_forced_complexity_ramp()
        _validate_scenario(s)
        assert s.name == "forced_complexity_ramp"
        assert s.category == "complexity_escalation"
        assert s.severity == 3

    def test_deep_nesting(self) -> None:
        s = self.cls.scenario_deep_nesting()
        _validate_scenario(s)
        assert s.name == "deep_nesting"
        assert s.category == "complexity_escalation"

    def test_long_function_body(self) -> None:
        s = self.cls.scenario_long_function_body()
        _validate_scenario(s)
        assert s.name == "long_function_body"

    def test_state_explosion(self) -> None:
        s = self.cls.scenario_state_explosion()
        _validate_scenario(s)
        assert s.name == "state_explosion"
        assert s.severity == 4
        assert s.recovery_expected is False

    def test_setup_callables_run(self, mock_agent) -> None:
        for method_name in [
            "scenario_forced_complexity_ramp",
            "scenario_deep_nesting",
            "scenario_long_function_body",
            "scenario_state_explosion",
        ]:
            s = getattr(self.cls, method_name)()
            s.setup(mock_agent)


# ------------------------------------------------------------------ #
# CircularDependencyScenarios
# ------------------------------------------------------------------ #


class TestCircularDependencyScenarios:
    def setup_method(self) -> None:
        self.cls = CircularDependencyScenarios()

    def test_mutual_recursion(self) -> None:
        s = self.cls.scenario_mutual_recursion()
        _validate_scenario(s)
        assert s.name == "mutual_recursion"
        assert s.category == "circular_dependency"
        assert s.severity == 3

    def test_indirect_cycle(self) -> None:
        s = self.cls.scenario_indirect_cycle()
        _validate_scenario(s)
        assert s.name == "indirect_cycle"
        assert s.category == "circular_dependency"
        assert s.severity == 4

    def test_setup_installs_functions(self, mock_agent) -> None:
        s = self.cls.scenario_mutual_recursion()
        s.setup(mock_agent)
        assert mock_agent.get_function_source("is_even") is not None
        assert mock_agent.get_function_source("is_odd") is not None

    def test_indirect_cycle_setup_installs_functions(self, mock_agent) -> None:
        s = self.cls.scenario_indirect_cycle()
        s.setup(mock_agent)
        assert mock_agent.get_function_source("parse_input") is not None
        assert mock_agent.get_function_source("transform_data") is not None
        assert mock_agent.get_function_source("format_output") is not None


# ------------------------------------------------------------------ #
# RollbackCorruptionScenarios
# ------------------------------------------------------------------ #


class TestRollbackCorruptionScenarios:
    def setup_method(self) -> None:
        self.cls = RollbackCorruptionScenarios()

    def test_corrupt_checkpoint_format(self) -> None:
        s = self.cls.scenario_corrupt_checkpoint_format()
        _validate_scenario(s)
        assert s.name == "corrupt_checkpoint_format"
        assert s.category == "rollback_corruption"
        assert s.severity == 4

    def test_corrupt_baseline_score(self) -> None:
        s = self.cls.scenario_corrupt_baseline_score()
        _validate_scenario(s)
        assert s.name == "corrupt_baseline_score"
        assert s.severity == 5
        assert s.recovery_expected is False

    def test_slow_rollback(self) -> None:
        s = self.cls.scenario_slow_rollback()
        _validate_scenario(s)
        assert s.name == "slow_rollback"
        assert s.severity == 2
        assert s.recovery_expected is True

    def test_setup_callables_run(self, mock_agent) -> None:
        for method_name in [
            "scenario_corrupt_checkpoint_format",
            "scenario_corrupt_baseline_score",
            "scenario_slow_rollback",
        ]:
            s = getattr(self.cls, method_name)()
            s.setup(mock_agent)


# ------------------------------------------------------------------ #
# SelfReferenceAttacks
# ------------------------------------------------------------------ #


class TestSelfReferenceAttacks:
    def setup_method(self) -> None:
        self.cls = SelfReferenceAttacks()

    def test_modify_modifier(self) -> None:
        s = self.cls.scenario_modify_modifier()
        _validate_scenario(s)
        assert s.name == "modify_modifier"
        assert s.category == "self_reference"
        assert s.severity == 4

    def test_modify_validation(self) -> None:
        s = self.cls.scenario_modify_validation()
        _validate_scenario(s)
        assert s.name == "modify_validation"
        assert s.severity == 5
        assert s.recovery_expected is False

    def test_nested_self_modification(self) -> None:
        s = self.cls.scenario_nested_self_modification()
        _validate_scenario(s)
        assert s.name == "nested_self_modification"
        assert s.severity == 5

    def test_infinite_modification_loop(self) -> None:
        s = self.cls.scenario_infinite_modification_loop()
        _validate_scenario(s)
        assert s.name == "infinite_modification_loop"
        assert s.severity == 4

    def test_rollback_of_rollback(self) -> None:
        s = self.cls.scenario_rollback_of_rollback()
        _validate_scenario(s)
        assert s.name == "rollback_of_rollback"
        assert s.severity == 3

    def test_setup_callables_run(self, mock_agent) -> None:
        for method_name in [
            "scenario_modify_modifier",
            "scenario_modify_validation",
            "scenario_nested_self_modification",
            "scenario_infinite_modification_loop",
            "scenario_rollback_of_rollback",
        ]:
            s = getattr(self.cls, method_name)()
            s.setup(mock_agent)
