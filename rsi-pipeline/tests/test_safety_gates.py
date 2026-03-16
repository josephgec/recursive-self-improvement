"""Tests for safety gates: GDI, constraint enforcement, CAR, emergency stop."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.pipeline.state import PipelineState, AgentCodeSnapshot, PerformanceRecord, SafetyStatus
from src.safety.gdi_monitor import GDIMonitor
from src.safety.constraint_enforcer import ConstraintEnforcer, ConstraintVerdict
from src.safety.car_tracker import CARTracker
from src.safety.emergency_stop import EmergencyStop
from src.safety.human_checkpoint import HumanCheckpoint


def _make_state(code="def solve(x): return x + 1", accuracy=0.7, entropy=1.0):
    return PipelineState(
        agent_code=AgentCodeSnapshot(code=code),
        original_code=AgentCodeSnapshot(code=code),
        performance=PerformanceRecord(accuracy=accuracy, entropy=entropy),
    )


class TestGDIMonitor:
    """Test GDI monitor."""

    def test_identical_code_zero_drift(self):
        gdi = GDIMonitor()
        score = gdi.compute("def f(): pass", "def f(): pass")
        assert score == 0.0

    def test_different_code_positive_drift(self):
        gdi = GDIMonitor()
        score = gdi.compute("def f(): return 1", "def g(): return 2 + 3 + 4 + 5")
        assert score > 0.0

    def test_completely_different_code_high_drift(self):
        gdi = GDIMonitor()
        score = gdi.compute("alpha beta gamma", "x y z w v u t s r q p o n m l k j i h g f e d c b a")
        assert score > 0.3

    def test_empty_code_max_drift(self):
        gdi = GDIMonitor()
        score = gdi.compute("", "some code")
        assert score == 1.0

    def test_both_empty_zero_drift(self):
        gdi = GDIMonitor()
        score = gdi.compute("", "")
        assert score == 0.0

    def test_check_threshold_pass(self):
        gdi = GDIMonitor(threshold=0.5)
        assert gdi.check_threshold(0.3) is False  # below threshold = no violation

    def test_check_threshold_fail(self):
        gdi = GDIMonitor(threshold=0.3)
        assert gdi.check_threshold(0.5) is True  # above threshold = violation

    def test_history_tracked(self):
        gdi = GDIMonitor()
        gdi.compute("a", "b")
        gdi.compute("x", "y")
        assert len(gdi.history) == 2


class TestConstraintEnforcer:
    """Test constraint enforcement."""

    def test_all_satisfied(self):
        enforcer = ConstraintEnforcer(accuracy_floor=0.5, entropy_floor=0.05, drift_ceiling=0.8)
        state = _make_state(accuracy=0.7, entropy=0.5)
        verdict = enforcer.check_all(state)

        assert verdict.satisfied is True
        assert len(verdict.violations) == 0

    def test_accuracy_below_floor(self):
        enforcer = ConstraintEnforcer(accuracy_floor=0.8)
        state = _make_state(accuracy=0.5)
        verdict = enforcer.check_all(state)

        assert verdict.satisfied is False
        assert any("accuracy" in v for v in verdict.violations)

    def test_entropy_below_floor(self):
        enforcer = ConstraintEnforcer(entropy_floor=0.5)
        state = _make_state(entropy=0.01)
        verdict = enforcer.check_all(state)

        assert verdict.satisfied is False
        assert any("entropy" in v for v in verdict.violations)

    def test_drift_above_ceiling(self):
        enforcer = ConstraintEnforcer(drift_ceiling=0.2)
        state = _make_state()
        state.safety.gdi_score = 0.5
        verdict = enforcer.check_all(state)

        assert verdict.satisfied is False
        assert any("drift" in v for v in verdict.violations)

    def test_multiple_violations(self):
        enforcer = ConstraintEnforcer(accuracy_floor=0.9, entropy_floor=2.0)
        state = _make_state(accuracy=0.5, entropy=0.01)
        verdict = enforcer.check_all(state)

        assert verdict.satisfied is False
        assert len(verdict.violations) >= 2

    def test_verdict_to_dict(self):
        verdict = ConstraintVerdict(satisfied=False, violations=["test"])
        d = verdict.to_dict()
        assert d["satisfied"] is False
        assert "test" in d["violations"]

    def test_properties(self):
        enforcer = ConstraintEnforcer(accuracy_floor=0.6, entropy_floor=0.1, drift_ceiling=0.5)
        assert enforcer.accuracy_floor == 0.6
        assert enforcer.entropy_floor == 0.1
        assert enforcer.drift_ceiling == 0.5


class TestCARTracker:
    """Test CAR tracker."""

    def test_improvement_car_is_one(self):
        car = CARTracker()
        score = car.compute(0.7, 0.8)
        assert score == 1.0

    def test_no_change_car_is_one(self):
        car = CARTracker()
        score = car.compute(0.7, 0.7)
        assert score == 1.0

    def test_degradation_car_below_one(self):
        car = CARTracker()
        score = car.compute(0.8, 0.4)
        assert score < 1.0
        assert score == 0.5

    def test_severe_degradation_low_car(self):
        car = CARTracker()
        score = car.compute(1.0, 0.1)
        assert score < 0.5

    def test_zero_before_handled(self):
        car = CARTracker()
        score = car.compute(0.0, 0.5)
        assert score == 1.0

    def test_is_pareto_improvement_true(self):
        car = CARTracker()
        assert car.is_pareto_improvement(0.7, 0.8) is True

    def test_is_pareto_improvement_false(self):
        car = CARTracker()
        assert car.is_pareto_improvement(0.8, 0.7) is False

    def test_is_pareto_improvement_equal(self):
        car = CARTracker()
        assert car.is_pareto_improvement(0.7, 0.7) is True

    def test_history(self):
        car = CARTracker()
        car.compute(0.7, 0.8)
        car.compute(0.8, 0.6)
        assert len(car.history) == 2

    def test_average_car(self):
        car = CARTracker()
        car.compute(0.5, 0.5)  # 1.0
        car.compute(1.0, 0.5)  # 0.5
        avg = car.average_car()
        assert avg == 0.75

    def test_average_car_empty(self):
        car = CARTracker()
        assert car.average_car() == 1.0


class TestEmergencyStop:
    """Test emergency stop conditions."""

    def test_no_trigger_normal_state(self):
        estop = EmergencyStop()
        state = _make_state()
        assert estop.check(state) is False

    def test_trigger_on_low_car(self):
        estop = EmergencyStop(car_threshold=0.5)
        state = _make_state()
        state.safety.car_score = 0.3
        assert estop.check(state) is True
        assert estop.triggered is True
        assert "car" in estop.reason

    def test_trigger_on_consecutive_rollbacks(self):
        estop = EmergencyStop(max_consecutive_rollbacks=3)
        state = _make_state()
        state.safety.consecutive_rollbacks = 3
        assert estop.check(state) is True
        assert "rollbacks" in estop.reason

    def test_trigger_on_constraint_violation(self):
        estop = EmergencyStop()
        state = _make_state()
        state.safety.constraints_satisfied = False
        state.safety.violations = ["accuracy_below_floor"]
        assert estop.check(state) is True

    def test_execute(self):
        estop = EmergencyStop()
        state = _make_state()
        estop.execute(state, "test_reason")

        assert state.status == "emergency"
        assert state.safety.emergency_stop is True
        assert estop.triggered is True

    def test_reset(self):
        estop = EmergencyStop()
        state = _make_state()
        state.safety.car_score = 0.1
        estop.check(state)
        assert estop.triggered is True

        estop.reset()
        assert estop.triggered is False
        assert estop.reason == ""


class TestHumanCheckpoint:
    """Test human checkpoint."""

    def test_should_pause_on_interval(self):
        hcp = HumanCheckpoint(review_interval=5)
        state = _make_state()
        state.iteration = 10
        assert hcp.should_pause(state) is True

    def test_should_not_pause_between_intervals(self):
        hcp = HumanCheckpoint(review_interval=10)
        state = _make_state()
        state.iteration = 3
        assert hcp.should_pause(state) is False

    def test_should_pause_on_emergency(self):
        hcp = HumanCheckpoint(review_interval=100)
        state = _make_state()
        state.safety.emergency_stop = True
        assert hcp.should_pause(state) is True

    def test_present_review(self):
        hcp = HumanCheckpoint()
        state = _make_state()
        review = hcp.present_review(state)

        assert "iteration" in review
        assert "accuracy" in review
        assert "gdi_score" in review
        assert hcp.review_count == 1

    def test_auto_approve_enabled(self):
        hcp = HumanCheckpoint(auto_approve_mode=True)
        state = _make_state()
        assert hcp.auto_approve(state) is True

    def test_auto_approve_disabled(self):
        hcp = HumanCheckpoint(auto_approve_mode=False)
        state = _make_state()
        assert hcp.auto_approve(state) is False

    def test_auto_approve_with_reviewer(self):
        hcp = HumanCheckpoint(reviewer=lambda s: s.performance.accuracy > 0.5)
        state = _make_state(accuracy=0.7)
        assert hcp.auto_approve(state) is True

    def test_reviews_list(self):
        hcp = HumanCheckpoint()
        state = _make_state()
        hcp.present_review(state)
        hcp.present_review(state)
        assert len(hcp.reviews) == 2
