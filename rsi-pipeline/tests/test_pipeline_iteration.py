"""Tests for pipeline iteration results, improvement curve, and convergence."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.pipeline.state import PipelineState, AgentCodeSnapshot, PerformanceRecord, SafetyStatus
from src.pipeline.iteration import IterationResult, PipelineResult
from src.pipeline.config import PipelineConfig
from src.pipeline.lifecycle import PipelineLifecycle
from src.tracking.improvement_curve import ImprovementCurveTracker
from src.tracking.iteration_logger import IterationLogger
from src.tracking.dashboard import DashboardConfig, PanelConfig
from src.tracking.lineage_tracker import LineageTracker
from src.analysis.convergence import ConvergenceAnalyzer
from src.analysis.paradigm_contribution import ParadigmContributionAnalyzer
from src.analysis.safety_report import SafetyReportGenerator
from src.analysis.report import generate_report
from src.outer_loop.candidate_pool import CandidatePool
from src.outer_loop.strategy_evolver import Candidate
from src.outer_loop.population_bridge import PopulationBridge
from src.outer_loop.hindsight_adapter import HindsightAdapter


class TestIterationResult:
    """Test IterationResult dataclass."""

    def test_fields(self):
        r = IterationResult(
            iteration=5,
            improved=True,
            accuracy_before=0.7,
            accuracy_after=0.85,
            safety_verdict="pass",
        )
        assert r.iteration == 5
        assert r.improved is True
        assert r.accuracy_delta == pytest.approx(0.15)

    def test_to_dict(self):
        r = IterationResult(iteration=1)
        d = r.to_dict()
        assert "iteration" in d
        assert "improved" in d
        assert "safety_verdict" in d

    def test_defaults(self):
        r = IterationResult()
        assert r.iteration == 0
        assert r.improved is False
        assert r.error is None
        assert r.rolled_back is False


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_improvement_rate(self):
        r = PipelineResult(total_iterations=10, successful_improvements=3)
        assert r.improvement_rate == 0.3

    def test_improvement_rate_zero_iterations(self):
        r = PipelineResult(total_iterations=0)
        assert r.improvement_rate == 0.0

    def test_total_accuracy_gain(self):
        r = PipelineResult(initial_accuracy=0.5, final_accuracy=0.8)
        assert r.total_accuracy_gain == pytest.approx(0.3)

    def test_to_dict(self):
        r = PipelineResult(total_iterations=5)
        d = r.to_dict()
        assert "improvement_rate" in d
        assert "total_accuracy_gain" in d


class TestImprovementCurveTracker:
    """Test improvement curve tracking."""

    def test_record_and_retrieve(self):
        tracker = ImprovementCurveTracker()
        tracker.record(0.7, 0)
        tracker.record(0.75, 1)
        tracker.record(0.8, 2)

        curve = tracker.compute_curve()
        assert len(curve) == 3
        assert curve[0] == (0, 0.7)
        assert curve[2] == (2, 0.8)

    def test_detect_plateau(self):
        tracker = ImprovementCurveTracker(window_size=5, plateau_tolerance=0.01)
        for i in range(10):
            tracker.record(0.8, i)

        assert tracker.detect_plateau() is True

    def test_no_plateau_with_variation(self):
        tracker = ImprovementCurveTracker(window_size=5, plateau_tolerance=0.001)
        for i in range(10):
            tracker.record(0.5 + i * 0.05, i)

        assert tracker.detect_plateau() is False

    def test_detect_degradation(self):
        tracker = ImprovementCurveTracker()
        tracker.record(0.8, 0)
        tracker.record(0.75, 1)
        tracker.record(0.7, 2)

        assert tracker.detect_degradation() is True

    def test_no_degradation(self):
        tracker = ImprovementCurveTracker()
        tracker.record(0.7, 0)
        tracker.record(0.75, 1)
        tracker.record(0.8, 2)

        assert tracker.detect_degradation() is False

    def test_marginal_returns(self):
        tracker = ImprovementCurveTracker()
        tracker.record(0.5, 0)
        tracker.record(0.7, 1)
        tracker.record(0.72, 2)

        mr = tracker.marginal_returns()
        assert mr < 1.0  # diminishing returns

    def test_marginal_returns_insufficient_data(self):
        tracker = ImprovementCurveTracker()
        tracker.record(0.5, 0)
        assert tracker.marginal_returns() == 1.0

    def test_latest_accuracy(self):
        tracker = ImprovementCurveTracker()
        assert tracker.latest_accuracy is None
        tracker.record(0.7, 0)
        assert tracker.latest_accuracy == 0.7

    def test_size(self):
        tracker = ImprovementCurveTracker()
        assert tracker.size == 0
        tracker.record(0.7, 0)
        assert tracker.size == 1


class TestConvergenceAnalyzer:
    """Test convergence analysis."""

    def test_converged(self):
        analyzer = ConvergenceAnalyzer(window_size=5, threshold=0.01)
        curve = [(i, 0.8) for i in range(10)]
        assert analyzer.is_converged(curve) is True

    def test_not_converged(self):
        analyzer = ConvergenceAnalyzer(window_size=5, threshold=0.001)
        curve = [(i, 0.5 + i * 0.05) for i in range(10)]
        assert analyzer.is_converged(curve) is False

    def test_insufficient_data(self):
        analyzer = ConvergenceAnalyzer(window_size=10)
        curve = [(0, 0.5), (1, 0.6)]
        assert analyzer.is_converged(curve) is False

    def test_estimate_ceiling(self):
        analyzer = ConvergenceAnalyzer()
        curve = [(i, 0.5 + i * 0.05) for i in range(10)]
        ceiling = analyzer.estimate_ceiling(curve)
        assert ceiling >= 0.95  # should be close to max observed
        assert ceiling <= 1.0

    def test_estimate_ceiling_empty(self):
        analyzer = ConvergenceAnalyzer()
        assert analyzer.estimate_ceiling([]) == 0.0

    def test_estimate_ceiling_single(self):
        analyzer = ConvergenceAnalyzer()
        assert analyzer.estimate_ceiling([(0, 0.5)]) == 0.5

    def test_marginal_returns(self):
        analyzer = ConvergenceAnalyzer()
        curve = [(0, 0.5), (1, 0.6), (2, 0.65), (3, 0.66)]
        mr = analyzer.marginal_returns(curve)
        assert isinstance(mr, float)


class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_defaults(self):
        config = PipelineConfig()
        assert config.get("pipeline.max_iterations") == 100
        assert config.get("safety.gdi.threshold") == 0.3

    def test_get_missing_returns_default(self):
        config = PipelineConfig()
        assert config.get("nonexistent.key", 42) == 42

    def test_set(self):
        config = PipelineConfig()
        config.set("pipeline.max_iterations", 50)
        assert config.get("pipeline.max_iterations") == 50

    def test_merge_configs(self):
        base = PipelineConfig()
        override = PipelineConfig({"pipeline": {"max_iterations": 50}})
        merged = PipelineConfig.merge_configs(base, override)
        assert merged.get("pipeline.max_iterations") == 50
        # Base values preserved for non-overridden keys
        assert merged.get("safety.gdi.threshold") == 0.3

    def test_from_yaml(self, tmp_path):
        yaml_content = """pipeline:
  max_iterations: 25
safety:
  gdi:
    threshold: 0.4
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.get("pipeline.max_iterations") == 25
        assert config.get("safety.gdi.threshold") == 0.4

    def test_data_property(self):
        config = PipelineConfig()
        assert isinstance(config.data, dict)


class TestPipelineLifecycle:
    """Test pipeline lifecycle."""

    def test_start(self):
        lc = PipelineLifecycle()
        state = PipelineState()
        state = lc.start(state)
        assert state.status == "running"

    def test_pause(self):
        lc = PipelineLifecycle()
        state = PipelineState(status="running")
        state = lc.pause(state)
        assert state.status == "paused"

    def test_resume(self):
        lc = PipelineLifecycle()
        state = PipelineState(status="paused")
        state = lc.resume(state)
        assert state.status == "running"

    def test_resume_non_paused_raises(self):
        lc = PipelineLifecycle()
        state = PipelineState(status="running")
        with pytest.raises(ValueError):
            lc.resume(state)

    def test_stop(self):
        lc = PipelineLifecycle()
        state = PipelineState(status="running")
        state = lc.stop(state, "done")
        assert state.status == "stopped"

    def test_checkpoint_and_restore(self, tmp_path):
        lc = PipelineLifecycle(checkpoint_dir=str(tmp_path))
        state = PipelineState(
            iteration=5,
            agent_code=AgentCodeSnapshot(code="test code"),
        )
        path = lc.checkpoint(state)
        assert os.path.exists(path)

        restored = lc.restore(path)
        assert restored.iteration == 5
        assert restored.agent_code.code == "test code"

    def test_checkpoints_list(self, tmp_path):
        lc = PipelineLifecycle(checkpoint_dir=str(tmp_path))
        state = PipelineState()
        lc.checkpoint(state)
        assert len(lc.checkpoints) == 1


class TestPipelineState:
    """Test pipeline state."""

    def test_state_id_changes_with_code(self):
        s1 = PipelineState(agent_code=AgentCodeSnapshot(code="a"))
        s2 = PipelineState(agent_code=AgentCodeSnapshot(code="b"))
        assert s1.state_id != s2.state_id

    def test_serialization_roundtrip(self):
        state = PipelineState(
            iteration=10,
            agent_code=AgentCodeSnapshot(code="code", version=3),
            performance=PerformanceRecord(accuracy=0.85),
        )
        json_str = state.to_json()
        restored = PipelineState.from_json(json_str)
        assert restored.iteration == 10
        assert restored.agent_code.code == "code"
        assert restored.performance.accuracy == 0.85

    def test_save_and_load(self, tmp_path):
        state = PipelineState(iteration=7)
        path = str(tmp_path / "state.json")
        state.save(path)
        loaded = PipelineState.load(path)
        assert loaded.iteration == 7


class TestCandidatePool:
    """Test candidate pool."""

    def test_add_and_size(self):
        pool = CandidatePool()
        pool.add(Candidate(candidate_id="a", score=0.5))
        pool.add(Candidate(candidate_id="b", score=0.9))
        assert pool.size == 2

    def test_get_best(self):
        pool = CandidatePool()
        pool.add(Candidate(candidate_id="a", score=0.5))
        pool.add(Candidate(candidate_id="b", score=0.9))
        pool.add(Candidate(candidate_id="c", score=0.7))

        best = pool.get_best(2)
        assert len(best) == 2
        assert best[0].score == 0.9

    def test_get_diverse(self):
        pool = CandidatePool()
        pool.add(Candidate(candidate_id="a", target="t1", operator="mutate"))
        pool.add(Candidate(candidate_id="b", target="t1", operator="mutate"))
        pool.add(Candidate(candidate_id="c", target="t2", operator="crossover"))

        diverse = pool.get_diverse(2)
        assert len(diverse) == 2
        targets = {c.target for c in diverse}
        assert len(targets) == 2

    def test_clear(self):
        pool = CandidatePool()
        pool.add(Candidate(candidate_id="a"))
        pool.clear()
        assert pool.size == 0

    def test_add_many(self):
        pool = CandidatePool()
        pool.add_many([Candidate(candidate_id="a"), Candidate(candidate_id="b")])
        assert pool.size == 2

    def test_candidates_property(self):
        pool = CandidatePool()
        c = Candidate(candidate_id="a")
        pool.add(c)
        assert len(pool.candidates) == 1


class TestPopulationBridge:
    """Test population bridge."""

    def test_sync(self):
        bridge = PopulationBridge(population_size=5)
        state = PipelineState(performance=PerformanceRecord(accuracy=0.7))
        bridge.sync(state)
        assert bridge.size == 1

    def test_size_limit(self):
        bridge = PopulationBridge(population_size=3)
        for i in range(10):
            state = PipelineState(
                iteration=i,
                performance=PerformanceRecord(accuracy=0.5 + i * 0.05),
            )
            bridge.sync(state)
        assert bridge.size <= 3

    def test_get_best(self):
        bridge = PopulationBridge()
        for acc in [0.5, 0.8, 0.6]:
            state = PipelineState(performance=PerformanceRecord(accuracy=acc))
            bridge.sync(state)
        best = bridge.get_best(1)
        assert best[0]["accuracy"] == 0.8

    def test_clear(self):
        bridge = PopulationBridge()
        state = PipelineState()
        bridge.sync(state)
        bridge.clear()
        assert bridge.size == 0


class TestHindsightAdapter:
    """Test hindsight adapter."""

    def test_collect(self):
        adapter = HindsightAdapter()
        result = IterationResult(iteration=1, improved=True)
        adapter.collect_from_iteration(result)
        assert adapter.pair_count == 1

    def test_feed_to_soar(self):
        adapter = HindsightAdapter()
        result = IterationResult(iteration=1, improved=True)
        adapter.collect_from_iteration(result)
        pairs = adapter.feed_to_soar()
        assert len(pairs) == 1
        assert pairs[0].reward == 1.0  # improved

    def test_negative_reward_on_failure(self):
        adapter = HindsightAdapter()
        result = IterationResult(iteration=1, improved=False)
        adapter.collect_from_iteration(result)
        pairs = adapter.feed_to_soar()
        assert pairs[0].reward == -0.5

    def test_clear(self):
        adapter = HindsightAdapter()
        result = IterationResult(iteration=1)
        adapter.collect_from_iteration(result)
        adapter.clear()
        assert adapter.pair_count == 0
        assert len(adapter.history) == 0


class TestIterationLogger:
    """Test iteration logger."""

    def test_log_and_retrieve(self):
        logger = IterationLogger()
        result = IterationResult(iteration=1)
        logger.log_iteration(result)
        assert logger.count == 1
        history = logger.get_history()
        assert len(history) == 1

    def test_get_last(self):
        logger = IterationLogger()
        for i in range(5):
            logger.log_iteration(IterationResult(iteration=i))
        last = logger.get_last(2)
        assert len(last) == 2

    def test_export_json(self):
        logger = IterationLogger()
        logger.log_iteration(IterationResult(iteration=0))
        json_str = logger.export("json")
        assert "iteration" in json_str

    def test_export_text(self):
        logger = IterationLogger()
        logger.log_iteration(IterationResult(iteration=0))
        text = logger.export("text")
        assert len(text) > 0

    def test_clear(self):
        logger = IterationLogger()
        logger.log_iteration(IterationResult())
        logger.clear()
        assert logger.count == 0


class TestDashboard:
    """Test dashboard config."""

    def test_default_panels(self):
        dashboard = DashboardConfig()
        panels = dashboard.get_panels()
        assert len(panels) >= 5
        names = [p.name for p in panels]
        assert "accuracy_curve" in names
        assert "gdi_gauge" in names

    def test_add_panel(self):
        dashboard = DashboardConfig()
        initial = len(dashboard.get_panels())
        dashboard.add_panel(PanelConfig(name="custom"))
        assert len(dashboard.get_panels()) == initial + 1

    def test_remove_panel(self):
        dashboard = DashboardConfig()
        initial = len(dashboard.get_panels())
        removed = dashboard.remove_panel("accuracy_curve")
        assert removed is True
        assert len(dashboard.get_panels()) == initial - 1

    def test_remove_nonexistent(self):
        dashboard = DashboardConfig()
        removed = dashboard.remove_panel("nonexistent")
        assert removed is False


class TestLineageTracker:
    """Test lineage tracker."""

    def test_record_and_get(self):
        tracker = LineageTracker()
        tracker.record_modification(0, "c1", "t", accuracy_before=0.7, accuracy_after=0.8)
        assert tracker.size == 1
        lineage = tracker.get_lineage()
        assert lineage[0]["improved"] is True

    def test_get_by_candidate(self):
        tracker = LineageTracker()
        tracker.record_modification(0, "c1", "t")
        tracker.record_modification(1, "c2", "t")
        lineage = tracker.get_lineage("c1")
        assert len(lineage) == 1

    def test_trace_improvement(self):
        tracker = LineageTracker()
        tracker.record_modification(0, "c1", "t", parent_ids=[])
        tracker.record_modification(1, "c2", "t", parent_ids=["c1"])
        chain = tracker.trace_improvement("c2")
        assert len(chain) >= 1

    def test_clear(self):
        tracker = LineageTracker()
        tracker.record_modification(0, "c1", "t")
        tracker.clear()
        assert tracker.size == 0


class TestParadigmContribution:
    """Test paradigm contribution analyzer."""

    def test_soar_efficiency(self):
        analyzer = ParadigmContributionAnalyzer()
        analyzer.set_modification_log([
            {"result": "applied"}, {"result": "rejected"}, {"result": "applied"},
        ])
        eff = analyzer.soar_efficiency()
        assert eff["total"] == 3
        assert eff["successful"] == 2
        assert eff["efficiency"] == pytest.approx(2 / 3)

    def test_soar_efficiency_empty(self):
        analyzer = ParadigmContributionAnalyzer()
        eff = analyzer.soar_efficiency()
        assert eff["efficiency"] == 0.0

    def test_verification_breakdown(self):
        analyzer = ParadigmContributionAnalyzer()
        analyzer.set_verification_log([
            {"empirical_passed": True, "compactness_passed": True},
            {"empirical_passed": False, "compactness_passed": True},
        ])
        bd = analyzer.verification_breakdown()
        assert bd["empirical_pass"] == 1
        assert bd["empirical_fail"] == 1
        assert bd["total"] == 2

    def test_modification_success_rate(self):
        analyzer = ParadigmContributionAnalyzer()
        analyzer.set_modification_log([{"result": "applied"}, {"result": "rejected"}])
        assert analyzer.modification_success_rate() == 0.5

    def test_plot_contribution_breakdown(self):
        analyzer = ParadigmContributionAnalyzer()
        result = analyzer.plot_contribution_breakdown()
        assert "soar" in result
        assert "paradigms" in result


class TestSafetyReport:
    """Test safety report generator."""

    def test_generate_report(self):
        gen = SafetyReportGenerator()
        gen.set_gdi_history([0.1, 0.15, 0.2])
        gen.set_car_history([1.0, 0.95, 0.9])

        state = PipelineState(
            safety=SafetyStatus(gdi_score=0.2, car_score=0.9),
            performance=PerformanceRecord(accuracy=0.8),
        )
        report = gen.generate_safety_report(state)

        assert "current_status" in report
        assert "gdi_trajectory" in report
        assert "car_trajectory" in report
        assert "risk_assessment" in report

    def test_gdi_trajectory(self):
        gen = SafetyReportGenerator()
        gen.set_gdi_history([0.1, 0.2, 0.3])
        traj = gen.gdi_trajectory()
        assert traj["trend"] == "increasing"
        assert traj["max"] == 0.3

    def test_car_trajectory(self):
        gen = SafetyReportGenerator()
        gen.set_car_history([1.0, 0.8, 0.6])
        traj = gen.car_trajectory()
        assert traj["trend"] == "decreasing"
        assert traj["min"] == 0.6

    def test_empty_trajectories(self):
        gen = SafetyReportGenerator()
        assert gen.gdi_trajectory()["trend"] == "stable"
        assert gen.car_trajectory()["trend"] == "stable"

    def test_add_violation(self):
        gen = SafetyReportGenerator()
        gen.add_violation({"type": "accuracy", "value": 0.4})
        state = PipelineState()
        report = gen.generate_safety_report(state)
        assert report["violation_summary"]["total"] == 1


class TestReportGeneration:
    """Test comprehensive report generation."""

    def test_generate_report_all_sections(self):
        report = generate_report(
            pipeline_result={
                "total_iterations": 10,
                "successful_improvements": 3,
                "rollbacks": 1,
                "emergency_stops": 0,
                "final_accuracy": 0.85,
                "initial_accuracy": 0.7,
                "total_accuracy_gain": 0.15,
                "improvement_rate": 0.3,
                "reason_stopped": "max_iterations",
                "iteration_results": [
                    {"iteration": 0, "improved": True, "accuracy_before": 0.7,
                     "accuracy_after": 0.75, "safety_verdict": "pass"},
                ],
            },
            safety_report={
                "current_status": {"gdi_score": 0.1, "car_score": 1.0,
                                   "constraints_satisfied": True, "consecutive_rollbacks": 0,
                                   "emergency_stop": False},
                "gdi_trajectory": {"trend": "stable", "current": 0.1, "max": 0.15, "avg": 0.1},
                "car_trajectory": {"trend": "stable", "current": 1.0, "min": 0.9, "avg": 0.95},
                "violation_summary": {"total": 0, "violations": []},
                "risk_assessment": "low",
            },
            convergence={"converged": False, "ceiling": 0.9, "marginal_returns": 0.7},
            paradigm_contribution={"soar": {"efficiency": 0.6}, "success_rate": 0.5,
                                   "verification": {"empirical_pass": 5, "empirical_fail": 2,
                                                    "compactness_pass": 4, "compactness_fail": 3}},
            improvement_curve=[(0, 0.7), (1, 0.75), (2, 0.8)],
            lineage=[{"candidate_id": "c1", "improved": True}],
        )

        # Check all 12 sections present
        assert "Executive Summary" in report
        assert "Pipeline Configuration" in report
        assert "Iteration Summary" in report
        assert "Improvement Curve" in report
        assert "Convergence Analysis" in report
        assert "Safety Status" in report
        assert "GDI Trajectory" in report
        assert "CAR Trajectory" in report
        assert "Modification History" in report
        assert "Paradigm Contributions" in report
        assert "Lineage Analysis" in report
        assert "Recommendations" in report

    def test_generate_report_empty(self):
        report = generate_report()
        assert "Executive Summary" in report
        assert "RSI Pipeline Report" in report

    def test_generate_report_critical_risk(self):
        report = generate_report(
            safety_report={"risk_assessment": "critical",
                           "current_status": {"emergency_stop": True},
                           "gdi_trajectory": {}, "car_trajectory": {},
                           "violation_summary": {"total": 0}},
        )
        assert "CRITICAL" in report
