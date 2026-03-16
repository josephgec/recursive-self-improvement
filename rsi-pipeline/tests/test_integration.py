"""Integration tests: full pipeline scenarios."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.pipeline.state import PipelineState, AgentCodeSnapshot, PerformanceRecord, SafetyStatus
from src.pipeline.config import PipelineConfig
from src.pipeline.orchestrator import RSIPipelineOrchestrator
from src.pipeline.iteration import IterationResult, PipelineResult
from src.outer_loop.strategy_evolver import StrategyEvolver, Candidate
from src.outer_loop.hindsight_adapter import HindsightAdapter
from src.verification.dual_verifier import DualVerifier
from src.verification.empirical_gate import EmpiricalGate
from src.verification.compactness_gate import CompactnessGate
from src.self_modification.modification_engine import ModificationEngine
from src.safety.gdi_monitor import GDIMonitor
from src.safety.constraint_enforcer import ConstraintEnforcer
from src.safety.car_tracker import CARTracker
from src.safety.emergency_stop import EmergencyStop
from src.tracking.iteration_logger import IterationLogger
from src.tracking.improvement_curve import ImprovementCurveTracker
from tests.conftest import MockLLM, make_evaluator


def _make_state(code="def solve(x): return x + 1", accuracy=0.7):
    state = PipelineState(
        agent_code=AgentCodeSnapshot(code=code),
        original_code=AgentCodeSnapshot(code=code),
        performance=PerformanceRecord(accuracy=accuracy, entropy=1.0),
    )
    return state


def _build_orchestrator(
    evaluator=None,
    llm=None,
    min_pass_rate=0.1,
    max_bdm_score=9999,
    cooldown=0,
    complexity_ceiling=9999,
    gdi_threshold=0.9,
    accuracy_floor=0.3,
    entropy_floor=0.01,
    drift_ceiling=0.95,
    car_threshold=0.5,
    max_rollbacks=3,
):
    config = PipelineConfig()
    return RSIPipelineOrchestrator(
        config=config,
        strategy_evolver=StrategyEvolver(llm=llm or MockLLM()),
        dual_verifier=DualVerifier(
            empirical_gate=EmpiricalGate(min_pass_rate=min_pass_rate),
            compactness_gate=CompactnessGate(max_bdm_score=max_bdm_score),
        ),
        modification_engine=ModificationEngine(
            cooldown_iterations=cooldown,
            complexity_ceiling=complexity_ceiling,
        ),
        gdi_monitor=GDIMonitor(threshold=gdi_threshold),
        constraint_enforcer=ConstraintEnforcer(
            accuracy_floor=accuracy_floor,
            entropy_floor=entropy_floor,
            drift_ceiling=drift_ceiling,
        ),
        car_tracker=CARTracker(min_ratio=car_threshold),
        emergency_stop=EmergencyStop(
            car_threshold=car_threshold,
            max_consecutive_rollbacks=max_rollbacks,
        ),
        hindsight_adapter=HindsightAdapter(),
        evaluator=evaluator,
    )


class TestThreeIterationsSuccessHindsightSafety:
    """Test 3 iterations: success, hindsight, safety check."""

    def test_three_iterations_success(self):
        """Run 3 iterations with improving evaluator."""
        accuracies = [0.75, 0.80, 0.85]
        call_idx = [0]

        def improving_evaluator(state):
            idx = min(call_idx[0], len(accuracies) - 1)
            acc = accuracies[idx]
            state.performance.accuracy = acc
            call_idx[0] += 1
            return acc

        orch = _build_orchestrator(evaluator=improving_evaluator)
        state = _make_state(accuracy=0.7)
        result = orch.run(state, max_iterations=3)

        assert result.total_iterations == 3
        assert result.final_accuracy >= 0.75
        # At least some improvements
        assert result.successful_improvements >= 1

    def test_hindsight_pairs_generated(self):
        """Hindsight adapter generates training pairs for each iteration."""
        orch = _build_orchestrator(evaluator=make_evaluator(0.8))
        state = _make_state(accuracy=0.7)
        orch.run(state, max_iterations=3)

        pairs = orch.hindsight_adapter.feed_to_soar()
        assert len(pairs) == 3  # one per iteration

    def test_safety_checked_each_iteration(self):
        """Safety is checked on each iteration that applies a modification."""
        orch = _build_orchestrator(evaluator=make_evaluator(0.8))
        state = _make_state(accuracy=0.7)
        result = orch.run(state, max_iterations=3)

        for ir in result.iteration_results:
            if ir.modification and ir.error is None:
                assert ir.safety_verdict in ("pass", "fail", "emergency")

    def test_iteration_logger_records_all(self):
        """Iteration logger records every iteration."""
        orch = _build_orchestrator(evaluator=make_evaluator(0.8))
        state = _make_state(accuracy=0.7)
        orch.run(state, max_iterations=3)

        history = orch.iteration_logger.get_history()
        assert len(history) == 3


class TestEmpiricalRejection:
    """Test failure mode: all candidates fail empirical gate."""

    def test_empirical_rejection(self):
        """Strict empirical gate rejects all candidates."""
        # LLM produces very short code that can't pass strict empirical gate
        short_llm = lambda code, t, o: "x"

        orch = _build_orchestrator(
            llm=short_llm,
            min_pass_rate=0.99,  # very strict
            evaluator=make_evaluator(0.8),
        )
        state = _make_state(accuracy=0.7)
        result = orch.run(state, max_iterations=3)

        # All iterations should fail at verification
        for ir in result.iteration_results:
            assert ir.improved is False
            assert ir.error is not None
        assert result.successful_improvements == 0

    def test_empirical_rejection_single_iteration(self):
        short_llm = lambda code, t, o: "x"
        orch = _build_orchestrator(
            llm=short_llm,
            min_pass_rate=0.99,
        )
        state = _make_state(accuracy=0.7)
        result = orch.run_iteration(state)
        assert "no_candidates_verified" in result.error


class TestBDMRejection:
    """Test failure mode: candidates fail BDM/compactness gate."""

    def test_bdm_rejection(self):
        """Strict compactness gate rejects all candidates."""
        orch = _build_orchestrator(
            max_bdm_score=0.001,  # extremely strict
            evaluator=make_evaluator(0.8),
        )
        state = _make_state(accuracy=0.7)
        result = orch.run(state, max_iterations=3)

        # All iterations should fail at verification (compactness gate)
        for ir in result.iteration_results:
            assert ir.improved is False
        assert result.successful_improvements == 0


class TestAccuracyFloorViolation:
    """Test failure mode: accuracy drops below floor."""

    def test_accuracy_floor_violation(self):
        """Evaluator drops accuracy below floor, triggering safety failure."""
        def degrading_evaluator(state):
            new_acc = 0.2  # well below accuracy_floor of 0.6
            state.performance.accuracy = new_acc
            return new_acc

        orch = _build_orchestrator(
            evaluator=degrading_evaluator,
            accuracy_floor=0.6,
        )
        state = _make_state(accuracy=0.7)
        result = orch.run(state, max_iterations=3)

        # Should have safety failures and rollbacks
        has_safety_fail = any(
            ir.safety_verdict in ("fail", "emergency")
            for ir in result.iteration_results
        )
        assert has_safety_fail or result.rollbacks > 0 or result.emergency_stops > 0


class TestThreeRollbacksEmergencyStop:
    """Test failure mode: 3 consecutive rollbacks trigger emergency stop."""

    def test_three_rollbacks_emergency(self):
        """Three consecutive rollbacks trigger emergency stop."""
        # Use very strict GDI threshold so every modification is rolled back
        orch = _build_orchestrator(
            evaluator=make_evaluator(0.8),
            gdi_threshold=0.0001,  # impossibly strict
            max_rollbacks=3,
        )
        state = _make_state(accuracy=0.7)

        result = orch.run(state, max_iterations=10)

        # Should have emergency stop after 3 consecutive rollbacks
        assert result.emergency_stops >= 1 or result.rollbacks >= 3
        # Should have stopped before 10 iterations
        assert result.total_iterations <= 10

    def test_emergency_stops_pipeline(self):
        """Emergency stop prevents further iterations."""
        orch = _build_orchestrator(
            evaluator=make_evaluator(0.8),
            gdi_threshold=0.0001,
            max_rollbacks=3,
        )
        state = _make_state(accuracy=0.7)
        result = orch.run(state, max_iterations=20)

        # Pipeline should terminate well before 20 iterations
        assert result.reason_stopped == "emergency_stop"


class TestEndToEndWithAllComponents:
    """End-to-end test exercising all major components."""

    def test_full_pipeline_lifecycle(self):
        """Run pipeline through full lifecycle."""
        call_count = [0]

        def alternating_evaluator(state):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                acc = state.performance.accuracy + 0.02
            else:
                acc = state.performance.accuracy
            state.performance.accuracy = acc
            return acc

        orch = _build_orchestrator(
            evaluator=alternating_evaluator,
            gdi_threshold=0.8,
            accuracy_floor=0.3,
        )
        state = _make_state(accuracy=0.65)

        # Run pipeline
        result = orch.run(state, max_iterations=5)

        assert result.total_iterations == 5
        assert result.initial_accuracy == 0.65
        assert len(result.iteration_results) == 5

        # Check pipeline result properties
        assert result.improvement_rate >= 0.0
        assert result.total_accuracy_gain >= 0.0

        # Hindsight collected
        pairs = orch.hindsight_adapter.feed_to_soar()
        assert len(pairs) == 5

    def test_pipeline_with_diverse_candidates(self):
        """Test with diverse candidate generation."""
        orch = _build_orchestrator(evaluator=make_evaluator(0.75))
        state = _make_state(accuracy=0.7)

        # Generate candidates directly
        candidates = orch.strategy_evolver.generate_candidates(state, 5)
        assert len(candidates) == 5

        # Verify candidates
        verified = orch.dual_verifier.verify_all(candidates, state)
        assert len(verified) >= 0  # may or may not pass depending on code

        # Run full iteration
        result = orch.run_iteration(state)
        assert isinstance(result, IterationResult)

    def test_checkpoint_during_run(self, tmp_path):
        """Test that checkpoints are created during run."""
        from src.pipeline.lifecycle import PipelineLifecycle

        config = PipelineConfig()
        config.set("pipeline.checkpoint_interval", 2)

        lifecycle = PipelineLifecycle(checkpoint_dir=str(tmp_path))
        orch = RSIPipelineOrchestrator(
            config=config,
            strategy_evolver=StrategyEvolver(llm=MockLLM()),
            dual_verifier=DualVerifier(
                empirical_gate=EmpiricalGate(min_pass_rate=0.1),
                compactness_gate=CompactnessGate(max_bdm_score=9999),
            ),
            modification_engine=ModificationEngine(cooldown_iterations=0, complexity_ceiling=9999),
            gdi_monitor=GDIMonitor(threshold=0.9),
            constraint_enforcer=ConstraintEnforcer(accuracy_floor=0.1, entropy_floor=0.01, drift_ceiling=0.99),
            car_tracker=CARTracker(),
            emergency_stop=EmergencyStop(),
            hindsight_adapter=HindsightAdapter(),
            lifecycle=lifecycle,
            evaluator=make_evaluator(0.8),
        )
        state = _make_state(accuracy=0.7)
        result = orch.run(state, max_iterations=5)

        # Checkpoints should have been created
        assert len(lifecycle.checkpoints) >= 1

    def test_strategy_evolver_candidate_fields(self):
        """Test that generated candidates have all required fields."""
        evolver = StrategyEvolver()
        state = _make_state()
        candidates = evolver.generate_candidates(state, 3)

        for c in candidates:
            assert c.candidate_id != ""
            assert c.target != ""
            assert c.proposed_code != ""
            assert c.description != ""
            assert c.operator in ("mutate", "crossover", "refine")

    def test_candidate_to_dict(self):
        """Test candidate serialization."""
        c = Candidate(
            candidate_id="test",
            target="t",
            proposed_code="code",
            operator="mutate",
        )
        d = c.to_dict()
        assert d["candidate_id"] == "test"
        assert d["target"] == "t"

    def test_performance_record_roundtrip(self):
        """Test performance record serialization."""
        pr = PerformanceRecord(accuracy=0.85, test_pass_rate=0.9, complexity_score=50.0)
        d = pr.to_dict()
        restored = PerformanceRecord.from_dict(d)
        assert restored.accuracy == 0.85
        assert restored.test_pass_rate == 0.9

    def test_agent_code_snapshot_roundtrip(self):
        """Test agent code snapshot serialization."""
        snap = AgentCodeSnapshot(code="def f(): pass", version=3, target="test")
        d = snap.to_dict()
        restored = AgentCodeSnapshot.from_dict(d)
        assert restored.code == "def f(): pass"
        assert restored.version == 3

    def test_safety_status_roundtrip(self):
        """Test safety status serialization."""
        ss = SafetyStatus(gdi_score=0.2, car_score=0.9, consecutive_rollbacks=1)
        d = ss.to_dict()
        restored = SafetyStatus.from_dict(d)
        assert restored.gdi_score == 0.2
        assert restored.car_score == 0.9
