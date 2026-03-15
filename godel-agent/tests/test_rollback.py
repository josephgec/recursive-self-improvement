"""Tests for RollbackManager and GuardedModification."""

from __future__ import annotations

import pytest

from src.core.state import AgentState, StateManager
from src.validation.rollback import RollbackManager, GuardedModification


@pytest.fixture
def rollback_mgr(state_manager: StateManager) -> RollbackManager:
    return RollbackManager(state_manager)


class TestCheckpoint:
    def test_checkpoint_creates_file(
        self, rollback_mgr: RollbackManager, sample_state: AgentState
    ) -> None:
        path = rollback_mgr.checkpoint(sample_state)
        assert path.exists()
        assert rollback_mgr.checkpoint_count == 1

    def test_multiple_checkpoints(
        self, rollback_mgr: RollbackManager, sample_state: AgentState
    ) -> None:
        rollback_mgr.checkpoint(sample_state)
        sample_state.iteration = 6
        rollback_mgr.checkpoint(sample_state)
        assert rollback_mgr.checkpoint_count == 2


class TestRollback:
    def test_rollback_to_last_checkpoint(
        self, rollback_mgr: RollbackManager, sample_state: AgentState
    ) -> None:
        rollback_mgr.checkpoint(sample_state)
        original_iteration = sample_state.iteration

        # Modify state
        sample_state.iteration = 99
        sample_state.accuracy_history.append(0.99)

        # Rollback
        restored = rollback_mgr.rollback()
        assert restored is not None
        assert restored.iteration == original_iteration

    def test_rollback_no_checkpoint_returns_none(
        self, rollback_mgr: RollbackManager
    ) -> None:
        result = rollback_mgr.rollback()
        assert result is None

    def test_rollback_by_state_id(
        self, rollback_mgr: RollbackManager, state_manager: StateManager
    ) -> None:
        state1 = AgentState(iteration=1, system_prompt="first")
        state_manager.capture(state1)
        rollback_mgr.checkpoint(state1)

        state2 = AgentState(iteration=2, system_prompt="second")
        state_manager.capture(state2)
        rollback_mgr.checkpoint(state2)

        # Rollback to state1
        restored = rollback_mgr.rollback(state1.state_id)
        assert restored is not None
        assert restored.iteration == 1

    def test_rollback_if_failed_with_failure(
        self, rollback_mgr: RollbackManager, sample_state: AgentState
    ) -> None:
        rollback_mgr.checkpoint(sample_state)
        original_iter = sample_state.iteration

        sample_state.iteration = 99
        result = rollback_mgr.rollback_if_failed(sample_state, condition=True)
        assert result.iteration == original_iter

    def test_rollback_if_failed_without_failure(
        self, rollback_mgr: RollbackManager, sample_state: AgentState
    ) -> None:
        rollback_mgr.checkpoint(sample_state)
        sample_state.iteration = 99
        result = rollback_mgr.rollback_if_failed(sample_state, condition=False)
        assert result.iteration == 99  # Not rolled back


class TestGuardedModification:
    def test_success_prevents_rollback(
        self, rollback_mgr: RollbackManager, sample_state: AgentState
    ) -> None:
        with GuardedModification(rollback_mgr, sample_state) as guard:
            sample_state.iteration = 99
            guard.mark_success()
        # Should not have rolled back since we marked success
        assert rollback_mgr.checkpoint_count == 1

    def test_exception_triggers_rollback(
        self, rollback_mgr: RollbackManager, sample_state: AgentState
    ) -> None:
        original_iter = sample_state.iteration
        with pytest.raises(ValueError):
            with GuardedModification(rollback_mgr, sample_state) as guard:
                sample_state.iteration = 99
                raise ValueError("Something went wrong")
        # The rollback was attempted (checkpoint was used)
        assert rollback_mgr.checkpoint_count == 1

    def test_no_mark_success_triggers_rollback(
        self, rollback_mgr: RollbackManager, sample_state: AgentState
    ) -> None:
        with GuardedModification(rollback_mgr, sample_state) as guard:
            sample_state.iteration = 99
            # Don't call mark_success()
        # Rollback should have been called
        assert rollback_mgr.checkpoint_count == 1
