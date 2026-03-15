"""Tests for the controlled environment and MockAgent."""

from __future__ import annotations

import textwrap
from typing import Any

import pytest

from src.adversarial.scenario_registry import AdversarialScenario
from src.harness.controlled_env import ControlledEnvironment, MockAgent


# ------------------------------------------------------------------ #
# MockAgent core interface
# ------------------------------------------------------------------ #


class TestMockAgentCore:
    def test_default_state(self) -> None:
        agent = MockAgent(seed=42)
        assert agent.is_functional() is True
        assert agent.can_modify() is True
        assert agent.can_validate() is True
        assert agent.modification_count() == 0
        assert agent.modification_depth() == 0
        assert agent.get_accuracy() > 0

    def test_install_and_get_function(self) -> None:
        agent = MockAgent()
        source = "def add(a, b):\n    return a + b\n"
        agent.install_function("add", source)
        assert agent.get_function_source("add") == source

    def test_get_function_source_missing(self) -> None:
        agent = MockAgent()
        assert agent.get_function_source("nonexistent") is None

    def test_get_all_code(self) -> None:
        agent = MockAgent()
        code = agent.get_all_code()
        assert "solve" in code

    def test_modify_function_valid(self) -> None:
        agent = MockAgent()
        new_source = "def solve(x):\n    return x * 3\n"
        result = agent.modify_function("solve", new_source)
        assert result is True
        assert agent.modification_count() == 1
        assert agent.get_function_source("solve") == new_source

    def test_modify_function_syntax_error_rejected(self) -> None:
        agent = MockAgent()
        result = agent.modify_function("solve", "def broken(")
        assert result is False
        # Modification was attempted but rejected
        assert agent.modification_count() == 1

    def test_modify_function_disabled(self) -> None:
        agent = MockAgent()
        agent._can_modify = False
        result = agent.modify_function("solve", "def solve(x):\n    return x\n")
        assert result is False

    def test_rollback(self) -> None:
        agent = MockAgent(seed=42)
        original_accuracy = agent.get_accuracy()
        # Modify to create checkpoint
        agent.modify_function("solve", "def solve(x):\n    return x * 3\n")
        # Rollback
        result = agent.rollback()
        assert result is True
        # Accuracy should be restored
        assert agent.get_accuracy() == pytest.approx(original_accuracy)

    def test_rollback_no_checkpoints(self) -> None:
        agent = MockAgent()
        result = agent.rollback()
        assert result is False

    def test_rollback_corrupted(self) -> None:
        agent = MockAgent(seed=42)
        agent.modify_function("solve", "def solve(x):\n    return x * 3\n")
        agent.corrupt_checkpoint(mode="format")
        result = agent.rollback()
        assert result is True
        # Corrupted rollback reduces accuracy by 50%
        assert agent.get_accuracy() < 1.0

    def test_run_iteration(self) -> None:
        agent = MockAgent()
        accuracy = agent.run_iteration()
        assert 0.0 <= accuracy <= 1.0

    def test_get_snapshot(self) -> None:
        agent = MockAgent()
        snapshot = agent.get_snapshot()
        assert "code" in snapshot
        assert "accuracy" in snapshot
        assert "complexity" in snapshot
        assert "modification_count" in snapshot
        assert "functional" in snapshot

    def test_get_complexity(self) -> None:
        agent = MockAgent()
        complexity = agent.get_complexity()
        assert complexity > 0


# ------------------------------------------------------------------ #
# MockAgent setup helpers
# ------------------------------------------------------------------ #


class TestMockAgentSetup:
    def test_set_task(self) -> None:
        agent = MockAgent()
        agent.set_task("my task")
        assert agent._task == "my task"

    def test_set_scoring_mode_inverted(self) -> None:
        agent = MockAgent()
        agent.set_scoring_mode("inverted")
        # Run iteration returns inverted accuracy
        accuracy = agent.run_iteration()
        true_accuracy = agent.get_true_accuracy()
        assert accuracy == pytest.approx(1.0 - true_accuracy)

    def test_get_accuracy_inverted(self) -> None:
        agent = MockAgent()
        agent.set_scoring_mode("inverted")
        acc = agent.get_accuracy()
        true_acc = agent.get_true_accuracy()
        assert acc == pytest.approx(1.0 - true_acc)

    def test_distribution_shift(self) -> None:
        agent = MockAgent()
        agent.set_distribution_shift(shift_at_iteration=2)
        acc_before = agent.get_true_accuracy()
        agent.run_iteration()  # iteration 1
        agent.run_iteration()  # iteration 2 -- shift applied
        agent.run_iteration()  # iteration 3 -- shift applied again
        agent.run_iteration()  # iteration 4 -- shift applied again
        agent.run_iteration()  # iteration 5 -- shift applied again
        # Accuracy should have decreased after several shifts
        assert agent.get_true_accuracy() <= acc_before

    def test_poisoning_schedule(self) -> None:
        agent = MockAgent()
        agent.set_poisoning_schedule(
            start_fraction=0.0, end_fraction=0.5, ramp_iterations=5
        )
        initial_acc = agent.get_true_accuracy()
        for _ in range(20):
            agent.run_iteration()
        # Accuracy should decrease due to poisoning over many iterations
        assert agent.get_true_accuracy() <= initial_acc

    def test_complexity_schedule(self) -> None:
        agent = MockAgent()
        agent.set_complexity_schedule("linear", start=30, step=50)
        initial_acc = agent.get_true_accuracy()
        for _ in range(10):
            agent.run_iteration()
        # After several iterations, target complexity > 200, accuracy degrades
        assert agent.get_true_accuracy() <= initial_acc

    def test_mark_function_as_target(self) -> None:
        agent = MockAgent()
        agent.mark_function_as_target("solve")
        assert agent._target_function == "solve"

    def test_enable_nested_modification(self) -> None:
        agent = MockAgent()
        agent.enable_nested_modification()
        assert agent._nested_mod is True

    def test_create_circular_dependency(self) -> None:
        agent = MockAgent()
        agent.create_circular_dependency("a", "b")
        assert ("a", "b") in agent._circular_deps

    def test_enable_rollback_tracking(self) -> None:
        agent = MockAgent()
        agent.enable_rollback_tracking()
        assert agent._rollback_tracking is True

    def test_set_nesting_depth(self) -> None:
        agent = MockAgent()
        agent.set_nesting_depth(10)
        assert agent._nesting_depth == 10

    def test_set_function_length(self) -> None:
        agent = MockAgent()
        agent.set_function_length(500)
        assert agent._function_length == 500

    def test_set_state_variables(self) -> None:
        agent = MockAgent()
        agent.set_state_variables(12)
        assert agent._state_variables == 12

    def test_corrupt_baseline(self) -> None:
        agent = MockAgent()
        agent.corrupt_baseline(new_score=0.0)
        assert agent._baseline_corrupted is True
        assert agent._accuracy == 0.0

    def test_set_rollback_delay(self) -> None:
        agent = MockAgent()
        agent.set_rollback_delay(seconds=3.0)
        assert agent._rollback_delay == 3.0

    def test_set_task_difficulty(self) -> None:
        agent = MockAgent()
        agent.set_task_difficulty("impossible")
        assert agent._task_difficulty == "impossible"


# ------------------------------------------------------------------ #
# ControlledEnvironment
# ------------------------------------------------------------------ #


class TestControlledEnvironment:
    def test_create_fresh_agent(self) -> None:
        env = ControlledEnvironment(seed=42)
        agent = env.create_fresh_agent()
        assert isinstance(agent, MockAgent)
        assert agent.is_functional()

    def test_apply_scenario(self) -> None:
        env = ControlledEnvironment()

        def setup(agent: Any) -> None:
            agent.set_task("test task")

        scenario = AdversarialScenario(
            name="test",
            category="test",
            description="A test",
            severity=1,
            setup=setup,
            expected_failure_mode="STAGNATION",
        )
        agent = env.create_fresh_agent()
        env.apply_scenario(agent, scenario)
        assert agent._task == "test task"

    def test_run_iterations(self) -> None:
        env = ControlledEnvironment()

        def setup(agent: Any) -> None:
            agent.set_task("test task")

        def success_criteria(agent: Any) -> bool:
            return agent.get_accuracy() >= 0.5

        scenario = AdversarialScenario(
            name="test",
            category="test",
            description="A test",
            severity=1,
            setup=setup,
            expected_failure_mode="STAGNATION",
            max_iterations=5,
            success_criteria=success_criteria,
        )
        agent = env.create_fresh_agent()
        env.apply_scenario(agent, scenario)
        results = env.run_iterations(agent, scenario)
        assert len(results) == 5
        for r in results:
            assert "iteration" in r
            assert "accuracy" in r
            assert "complexity" in r
            assert "functional" in r

    def test_run_iterations_custom_max(self) -> None:
        env = ControlledEnvironment()

        def setup(agent: Any) -> None:
            pass

        scenario = AdversarialScenario(
            name="test",
            category="test",
            description="A test",
            severity=1,
            setup=setup,
            expected_failure_mode="STAGNATION",
            max_iterations=100,
        )
        agent = env.create_fresh_agent()
        results = env.run_iterations(agent, scenario, max_iterations=3)
        assert len(results) == 3

    def test_run_iterations_stops_on_dead_agent(self) -> None:
        env = ControlledEnvironment()

        def setup(agent: Any) -> None:
            pass

        scenario = AdversarialScenario(
            name="test",
            category="test",
            description="A test",
            severity=1,
            setup=setup,
            expected_failure_mode="STAGNATION",
            max_iterations=10,
        )
        agent = env.create_fresh_agent()
        agent._functional = False
        results = env.run_iterations(agent, scenario)
        # Should stop after first iteration since agent is not functional
        assert len(results) == 1
