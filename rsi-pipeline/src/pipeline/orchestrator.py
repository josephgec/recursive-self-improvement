"""RSI Pipeline Orchestrator: runs the 6-step self-improvement loop."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.pipeline.state import PipelineState, PerformanceRecord
from src.pipeline.iteration import IterationResult, PipelineResult
from src.pipeline.config import PipelineConfig
from src.pipeline.lifecycle import PipelineLifecycle
from src.outer_loop.strategy_evolver import StrategyEvolver
from src.verification.dual_verifier import DualVerifier
from src.self_modification.modification_engine import ModificationEngine
from src.safety.gdi_monitor import GDIMonitor
from src.safety.constraint_enforcer import ConstraintEnforcer
from src.safety.car_tracker import CARTracker
from src.safety.emergency_stop import EmergencyStop
from src.outer_loop.hindsight_adapter import HindsightAdapter
from src.tracking.iteration_logger import IterationLogger
from src.tracking.improvement_curve import ImprovementCurveTracker


class RSIPipelineOrchestrator:
    """Orchestrates the recursive self-improvement pipeline.

    Six-step iteration:
        1. Generate candidates (StrategyEvolver)
        2. Verify candidates (DualVerifier)
        3. Modify agent code (ModificationEngine)
        4. Evaluate post-modification performance
        5. Safety checks (GDI, constraints, CAR, emergency)
        6. Hindsight adaptation
    """

    def __init__(
        self,
        config: PipelineConfig,
        strategy_evolver: StrategyEvolver,
        dual_verifier: DualVerifier,
        modification_engine: ModificationEngine,
        gdi_monitor: GDIMonitor,
        constraint_enforcer: ConstraintEnforcer,
        car_tracker: CARTracker,
        emergency_stop: EmergencyStop,
        hindsight_adapter: HindsightAdapter,
        iteration_logger: Optional[IterationLogger] = None,
        improvement_tracker: Optional[ImprovementCurveTracker] = None,
        lifecycle: Optional[PipelineLifecycle] = None,
        evaluator: Any = None,
    ):
        self.config = config
        self.strategy_evolver = strategy_evolver
        self.dual_verifier = dual_verifier
        self.modification_engine = modification_engine
        self.gdi_monitor = gdi_monitor
        self.constraint_enforcer = constraint_enforcer
        self.car_tracker = car_tracker
        self.emergency_stop = emergency_stop
        self.hindsight_adapter = hindsight_adapter
        self.iteration_logger = iteration_logger or IterationLogger()
        self.improvement_tracker = improvement_tracker or ImprovementCurveTracker()
        self.lifecycle = lifecycle or PipelineLifecycle()
        self.evaluator = evaluator  # callable(state) -> float accuracy

    def run(self, state: PipelineState, max_iterations: Optional[int] = None) -> PipelineResult:
        """Run the pipeline for up to max_iterations."""
        max_iter = max_iterations or self.config.get("pipeline.max_iterations", 100)
        result = PipelineResult(initial_accuracy=state.performance.accuracy)

        # Don't override emergency/stopped state
        if state.status not in ("emergency", "stopped"):
            state = self.lifecycle.start(state)

        for i in range(max_iter):
            if state.status == "emergency":
                result.emergency_stops += 1
                result.reason_stopped = "emergency_stop"
                break

            if state.status == "paused":
                result.reason_stopped = "paused"
                break

            if state.status == "stopped":
                result.reason_stopped = "stopped"
                break

            iter_result = self.run_iteration(state)
            result.iteration_results.append(iter_result)
            result.total_iterations += 1

            if iter_result.improved:
                result.successful_improvements += 1
            if iter_result.rolled_back:
                result.rollbacks += 1
            if iter_result.safety_verdict == "emergency":
                result.emergency_stops += 1
                result.reason_stopped = "emergency_stop"
                break

            state.iteration += 1

            # checkpoint at configured intervals
            interval = self.config.get("pipeline.checkpoint_interval", 10)
            if state.iteration % interval == 0:
                self.lifecycle.checkpoint(state)

        result.final_accuracy = state.performance.accuracy
        if result.reason_stopped == "":
            result.reason_stopped = "max_iterations"
        self.lifecycle.stop(state, result.reason_stopped)
        return result

    def run_iteration(self, state: PipelineState) -> IterationResult:
        """Run a single 6-step iteration."""
        result = IterationResult(iteration=state.iteration)
        result.accuracy_before = state.performance.accuracy

        # Step 1: Generate candidates
        n = self.config.get("pipeline.candidates_per_iteration", 5)
        candidates = self.strategy_evolver.generate_candidates(state, n)
        if not candidates:
            result.error = "no_candidates_generated"
            self._log_and_track(result, state)
            return result

        # Step 2: Verify candidates
        verified = self.dual_verifier.verify_all(candidates, state)
        if not verified:
            result.error = "no_candidates_verified"
            self._log_and_track(result, state)
            return result

        # Step 3: Apply best modification
        best = verified[0]
        result.candidate = {"candidate_id": best.candidate.candidate_id, "target": best.candidate.target}
        mod_result = self.modification_engine.apply(best, state)
        result.modification = mod_result.to_dict()

        if not mod_result.applied:
            result.error = f"modification_rejected: {mod_result.reason}"
            self._log_and_track(result, state)
            return result

        # Step 4: Evaluate post-modification
        if self.evaluator:
            new_accuracy = self.evaluator(state)
        else:
            # Default: the modification itself provides accuracy
            new_accuracy = state.performance.accuracy
        result.accuracy_after = new_accuracy

        # Step 5: Safety checks
        safety_verdict = self._run_safety_checks(state, result)
        result.safety_verdict = safety_verdict

        if safety_verdict == "fail":
            # Rollback
            self.modification_engine.rollback(state)
            result.rolled_back = True
            result.accuracy_after = result.accuracy_before
            state.performance.accuracy = result.accuracy_before
            state.safety.consecutive_rollbacks += 1
            # Check if this rollback triggers emergency stop
            if self.emergency_stop.check(state, [result]):
                result.safety_verdict = "emergency"
                state.status = "emergency"
        elif safety_verdict == "emergency":
            self.modification_engine.rollback(state)
            result.rolled_back = True
            result.accuracy_after = result.accuracy_before
            state.performance.accuracy = result.accuracy_before
            state.status = "emergency"
        else:
            result.improved = result.accuracy_after > result.accuracy_before
            state.safety.consecutive_rollbacks = 0
            # Record performance
            perf = PerformanceRecord(
                accuracy=result.accuracy_after,
                iteration=state.iteration,
            )
            state.performance = perf
            state.performance_history.append(perf)

        # Step 6: Hindsight adaptation
        self.hindsight_adapter.collect_from_iteration(result)

        self._log_and_track(result, state)
        return result

    def _run_safety_checks(self, state: PipelineState, result: IterationResult) -> str:
        """Run all safety checks. Returns 'pass', 'fail', or 'emergency'."""
        # GDI check
        gdi = self.gdi_monitor.compute(state.agent_code.code, state.original_code.code)
        state.safety.gdi_score = gdi
        if self.gdi_monitor.check_threshold(gdi):
            state.safety.violations.append(f"gdi_exceeded:{gdi:.3f}")
            return "fail"

        # Constraint enforcement
        verdict = self.constraint_enforcer.check_all(state)
        if not verdict.satisfied:
            state.safety.constraints_satisfied = False
            state.safety.violations.extend(verdict.violations)
            return "fail"

        # CAR tracking
        car = self.car_tracker.compute(result.accuracy_before, result.accuracy_after)
        state.safety.car_score = car

        # Emergency stop
        recent = [r for r in [result]]
        if self.emergency_stop.check(state, recent):
            return "emergency"

        return "pass"

    def _log_and_track(self, result: IterationResult, state: PipelineState) -> None:
        """Log iteration and track improvement curve."""
        self.iteration_logger.log_iteration(result)
        self.improvement_tracker.record(result.accuracy_after, result.iteration)
