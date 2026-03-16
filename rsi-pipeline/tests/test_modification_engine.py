"""Tests for the Modification Engine."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.pipeline.state import PipelineState, AgentCodeSnapshot, PerformanceRecord
from src.outer_loop.strategy_evolver import Candidate
from src.verification.dual_verifier import VerifiedCandidate
from src.verification.empirical_gate import EmpiricalResult
from src.verification.compactness_gate import CompactnessResult
from src.self_modification.modification_engine import ModificationEngine, ModificationResult
from src.self_modification.target_registry import TargetRegistry
from src.self_modification.rollback_bridge import RollbackBridge
from src.self_modification.audit_bridge import AuditBridge
from src.self_modification.safety_gate import SafetyGate


def _make_state(code="def solve(x): return x + 1", accuracy=0.7):
    return PipelineState(
        agent_code=AgentCodeSnapshot(code=code, version=0),
        original_code=AgentCodeSnapshot(code=code, version=0),
        performance=PerformanceRecord(accuracy=accuracy, entropy=1.0),
    )


def _make_verified(target="strategy_evolver", code="def solve(x): return x * 2"):
    candidate = Candidate(
        candidate_id="test_mod_001",
        target=target,
        proposed_code=code,
        operator="mutate",
    )
    return VerifiedCandidate(
        candidate=candidate,
        empirical=EmpiricalResult(candidate_id=candidate.candidate_id, passed=True, pass_rate=0.9, accuracy=0.85),
        compactness=CompactnessResult(candidate_id=candidate.candidate_id, passed=True, bdm_score=30, code_length=50),
        combined_score=0.8,
    )


class TestApplyAllowedTarget:
    """Test applying modification to an allowed target."""

    def test_apply_succeeds(self):
        engine = ModificationEngine(cooldown_iterations=0)
        state = _make_state()
        verified = _make_verified()

        result = engine.apply(verified, state)

        assert result.applied is True
        assert result.candidate_id == "test_mod_001"
        assert state.agent_code.code == "def solve(x): return x * 2"
        assert state.agent_code.version == 1

    def test_apply_records_in_history(self):
        engine = ModificationEngine(cooldown_iterations=0)
        state = _make_state()
        verified = _make_verified()

        engine.apply(verified, state)

        assert len(state.modification_history) == 1
        assert state.modification_history[0]["candidate_id"] == "test_mod_001"

    def test_apply_logs_audit(self):
        engine = ModificationEngine(cooldown_iterations=0)
        state = _make_state()
        verified = _make_verified()

        engine.apply(verified, state)

        assert engine.audit.modification_count == 1

    def test_modification_result_to_dict(self):
        result = ModificationResult(applied=True, candidate_id="x", target="t")
        d = result.to_dict()
        assert d["applied"] is True
        assert d["candidate_id"] == "x"


class TestRejectForbiddenTarget:
    """Test rejecting modification of forbidden targets."""

    def test_forbidden_target_rejected(self):
        engine = ModificationEngine(cooldown_iterations=0)
        state = _make_state()
        verified = _make_verified(target="emergency_stop")

        result = engine.apply(verified, state)

        assert result.applied is False
        assert "forbidden" in result.reason

    def test_unknown_target_rejected(self):
        engine = ModificationEngine(cooldown_iterations=0)
        state = _make_state()
        verified = _make_verified(target="nonexistent_target")

        result = engine.apply(verified, state)

        assert result.applied is False
        assert "not_allowed" in result.reason


class TestCooldown:
    """Test cooldown enforcement."""

    def test_cooldown_blocks_modification(self):
        engine = ModificationEngine(cooldown_iterations=5)
        state = _make_state()
        state.iteration = 2
        verified = _make_verified()

        # First apply succeeds
        result1 = engine.apply(verified, state)
        assert result1.applied is True

        # Second apply blocked by cooldown (same iteration)
        result2 = engine.apply(verified, state)
        assert result2.applied is False
        assert "cooldown" in result2.reason

    def test_cooldown_allows_after_elapsed(self):
        engine = ModificationEngine(cooldown_iterations=2)
        state = _make_state()
        state.iteration = 0
        verified = _make_verified()

        result1 = engine.apply(verified, state)
        assert result1.applied is True

        state.iteration = 5  # well past cooldown
        v2 = _make_verified(code="def solve(x): return x + 10")
        result2 = engine.apply(v2, state)
        assert result2.applied is True


class TestComplexityCeiling:
    """Test complexity ceiling enforcement."""

    def test_over_complexity_rejected(self):
        engine = ModificationEngine(cooldown_iterations=0, complexity_ceiling=10)
        state = _make_state()
        verified = _make_verified(code="x" * 100)

        result = engine.apply(verified, state)

        assert result.applied is False
        assert "complexity" in result.reason


class TestAutoRollbackOnValidationFail:
    """Test auto-rollback when validator fails."""

    def test_validation_failure_rollback(self):
        def failing_validator(state):
            return False

        engine = ModificationEngine(cooldown_iterations=0, validator=failing_validator)
        state = _make_state()
        original_code = state.agent_code.code
        verified = _make_verified()

        result = engine.apply(verified, state)

        assert result.applied is False
        assert "validation_failed" in result.reason
        assert state.agent_code.code == original_code  # rolled back

    def test_validation_error_rollback(self):
        def error_validator(state):
            raise RuntimeError("Validation exploded")

        engine = ModificationEngine(cooldown_iterations=0, validator=error_validator)
        state = _make_state()
        original_code = state.agent_code.code
        verified = _make_verified()

        result = engine.apply(verified, state)

        assert result.applied is False
        assert "validation_error" in result.reason
        assert state.agent_code.code == original_code

    def test_manual_rollback(self):
        engine = ModificationEngine(cooldown_iterations=0)
        state = _make_state()
        original_code = state.agent_code.code
        verified = _make_verified()

        engine.apply(verified, state)
        assert state.agent_code.code != original_code

        success = engine.rollback(state)
        assert success is True
        assert state.agent_code.code == original_code


class TestTargetRegistry:
    """Test target registry directly."""

    def test_allowed_targets(self):
        registry = TargetRegistry()
        assert registry.is_allowed("strategy_evolver")
        assert not registry.is_allowed("emergency_stop")

    def test_forbidden_targets(self):
        registry = TargetRegistry()
        assert registry.is_forbidden("emergency_stop")
        assert registry.is_forbidden("constraint_enforcer")
        assert not registry.is_forbidden("strategy_evolver")

    def test_list_allowed(self):
        registry = TargetRegistry()
        allowed = registry.list_allowed()
        assert "strategy_evolver" in allowed
        assert "emergency_stop" not in allowed

    def test_list_forbidden(self):
        registry = TargetRegistry()
        forbidden = registry.list_forbidden()
        assert "emergency_stop" in forbidden

    def test_add_allowed(self):
        registry = TargetRegistry()
        registry.add_allowed("new_target")
        assert registry.is_allowed("new_target")

    def test_add_forbidden(self):
        registry = TargetRegistry()
        registry.add_forbidden("new_forbidden")
        assert registry.is_forbidden("new_forbidden")


class TestRollbackBridge:
    """Test rollback bridge directly."""

    def test_checkpoint_and_rollback(self):
        bridge = RollbackBridge()
        state = _make_state()

        bridge.checkpoint(state)
        assert bridge.get_checkpoint_depth() == 1

        state.agent_code.code = "modified"
        success = bridge.rollback(state)
        assert success is True
        assert state.agent_code.code == "def solve(x): return x + 1"

    def test_rollback_empty(self):
        bridge = RollbackBridge()
        state = _make_state()
        assert bridge.rollback(state) is False

    def test_clear(self):
        bridge = RollbackBridge()
        state = _make_state()
        bridge.checkpoint(state)
        bridge.clear()
        assert bridge.get_checkpoint_depth() == 0


class TestAuditBridge:
    """Test audit bridge directly."""

    def test_log_modification(self):
        audit = AuditBridge()
        audit.log_modification("c1", "target", "old", "new")
        assert audit.modification_count == 1

    def test_log_rollback(self):
        audit = AuditBridge()
        audit.log_rollback("test_reason", iteration=5)
        assert audit.rollback_count == 1

    def test_export_history(self):
        audit = AuditBridge()
        audit.log_modification("c1", "t", "old", "new")
        audit.log_rollback("reason")
        history = audit.export_history()
        assert history["total_modifications"] == 1
        assert history["total_rollbacks"] == 1

    def test_export_json(self):
        audit = AuditBridge()
        audit.log_modification("c1", "t", "old", "new")
        json_str = audit.export_json()
        assert "c1" in json_str


class TestSafetyGate:
    """Test pre-modification safety gate."""

    def test_allows_normal_modification(self):
        gate = SafetyGate()
        candidate = Candidate(target="t", proposed_code="x" * 50)
        state = _make_state()
        result = gate.check_pre_modification(candidate, state)
        assert result["allowed"] is True

    def test_blocks_emergency_state(self):
        gate = SafetyGate()
        candidate = Candidate(target="t", proposed_code="x" * 50)
        state = _make_state()
        state.status = "emergency"
        result = gate.check_pre_modification(candidate, state)
        assert result["allowed"] is False

    def test_blocks_too_many_rollbacks(self):
        gate = SafetyGate()
        candidate = Candidate(target="t", proposed_code="x" * 50)
        state = _make_state()
        state.safety.consecutive_rollbacks = 5
        result = gate.check_pre_modification(candidate, state)
        assert result["allowed"] is False

    def test_blocks_overly_complex_code(self):
        gate = SafetyGate(max_complexity=10)
        candidate = Candidate(target="t", proposed_code="x" * 100)
        state = _make_state()
        result = gate.check_pre_modification(candidate, state)
        assert result["allowed"] is False
