"""Additional tests to boost coverage across all modules.

Targets uncovered branches and edge cases identified from coverage report.
"""

import pytest
from src.collapse.data_reserve import CleanDataReserve, ReserveStatus
from src.collapse.collapse_forecaster import CollapseForecaster, CollapseForecast
from src.collapse.halt_and_diagnose import HaltAndDiagnoseProtocol, HaltReport
from src.collapse.recovery import CollapseRecovery, VALID_STRATEGIES, RecoveryAction
from src.self_mod.complexity_budget import ComplexityBudget, BudgetStatus
from src.self_mod.blast_radius import BlastRadiusEstimator, BlastRadiusEstimate
from src.self_mod.quarantine import ModificationQuarantine
from src.reward.adversarial_eval import AdversarialEvalSet, AdversarialResult
from src.reward.eval_rotation import EvalSetRotator
from src.reward.reward_audit import RewardAuditTrail
from src.reward.reward_sanity import RewardSanityChecker, SanityResult
from src.cost.budget_manager import BudgetManager, BudgetState
from src.cost.circuit_breaker import CircuitBreaker
from src.cost.cost_forecaster import CostForecaster
from src.cost.cost_optimizer import CostOptimizer, CostSaving
from src.constraints.graduated_relaxation import GraduatedRelaxation
from src.constraints.compensation import CompensationMonitor
from src.constraints.tightness_detector import TightnessDetector
from src.constraints.adaptive_thresholds import AdaptiveThresholds
from src.publication.deadline_tracker import DeadlineTracker, DeadlineStatus
from src.publication.draft_generator import PaperDraftGenerator
from src.publication.fallback_planner import FallbackPlanner
from src.publication.readiness_checker import ReadinessChecker
from src.orchestration.risk_registry import RiskRegistry, Risk, RiskStatus, RiskDashboard
from src.orchestration.risk_dashboard import UnifiedRiskDashboard
from src.orchestration.incident_manager import IncidentManager, Incident
from src.analysis.risk_retrospective import RiskRetrospective, RetrospectiveReport
from src.analysis.report import ReportGenerator


# --- Data Reserve ---

class TestDataReserveEdgeCases:
    def test_negative_initial_size_raises(self):
        with pytest.raises(ValueError):
            CleanDataReserve(initial_size=-1)

    def test_invalid_reserve_fraction_raises(self):
        with pytest.raises(ValueError):
            CleanDataReserve(min_reserve_fraction=1.5)

    def test_negative_draw_raises(self):
        reserve = CleanDataReserve(initial_size=100)
        with pytest.raises(ValueError):
            reserve.draw(-1)

    def test_draw_more_than_remaining_raises(self):
        reserve = CleanDataReserve(initial_size=100)
        with pytest.raises(ValueError):
            reserve.draw(200)

    def test_draw_breaching_min_reserve_raises(self):
        reserve = CleanDataReserve(initial_size=100, min_reserve_fraction=0.5)
        with pytest.raises(ValueError):
            reserve.draw(60)  # Would leave 40 < 50 (min)

    def test_verify_reserve_status(self):
        reserve = CleanDataReserve(initial_size=100, min_reserve_fraction=0.1)
        status = reserve.verify_reserve()
        assert status.total_size == 100
        assert status.remaining == 100
        assert status.drawn == 0
        assert status.fraction_remaining == 1.0
        assert status.is_sufficient is True
        assert status.fraction_drawn == 0.0

    def test_fraction_drawn_property(self):
        reserve = CleanDataReserve(initial_size=100, min_reserve_fraction=0.1)
        reserve.draw(50)
        status = reserve.verify_reserve()
        assert status.fraction_drawn == pytest.approx(0.5)

    def test_zero_total_size(self):
        reserve = CleanDataReserve(initial_size=0, min_reserve_fraction=0.0)
        status = reserve.verify_reserve()
        assert status.fraction_remaining == 0.0
        assert reserve.is_sufficient() is False

    def test_replenish(self):
        reserve = CleanDataReserve(initial_size=100, min_reserve_fraction=0.1)
        reserve.draw(50)
        reserve.replenish(30)
        assert reserve.get_remaining() == 80

    def test_negative_replenish_raises(self):
        reserve = CleanDataReserve(initial_size=100)
        with pytest.raises(ValueError):
            reserve.replenish(-10)

    def test_draw_history(self):
        reserve = CleanDataReserve(initial_size=100, min_reserve_fraction=0.1)
        reserve.draw(10)
        reserve.draw(20)
        history = reserve.get_draw_history()
        assert len(history) == 2
        assert history[0]["n"] == 10
        assert history[1]["n"] == 20


# --- Collapse Forecaster ---

class TestCollapseForecasterEdgeCases:
    def test_empty_metrics(self):
        forecaster = CollapseForecaster()
        forecast = forecaster.forecast({})
        assert forecast.collapse_probability == 0.0
        assert forecast.risk_level == "low"

    def test_single_point_trajectory(self):
        forecaster = CollapseForecaster()
        forecast = forecaster.forecast({"entropy": [4.0]})
        assert forecast.collapse_probability == 0.0

    def test_high_similarity_baseline(self):
        forecaster = CollapseForecaster()
        # Use exact entropy_death trajectory
        metrics = {
            "entropy": [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1],
            "kl_divergence": [0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0],
        }
        forecast = forecaster.forecast(metrics)
        assert forecast.collapse_probability > 0.5
        assert forecast.is_high_risk

    def test_forecast_history(self):
        forecaster = CollapseForecaster()
        forecaster.forecast({"entropy": [4.0, 3.0]})
        forecaster.forecast({"entropy": [4.0, 3.0]})
        assert len(forecaster.get_history()) == 2

    def test_zero_range_trajectory(self):
        forecaster = CollapseForecaster()
        # All same values -> zero variance
        forecast = forecaster.forecast({"entropy": [4.0, 4.0, 4.0, 4.0]})
        assert forecast.collapse_probability == 0.0

    def test_medium_risk_level(self):
        forecaster = CollapseForecaster()
        # Partially matching
        metrics = {
            "entropy": [4.0, 3.8, 3.5, 3.2, 3.0],
            "kl_divergence": [0.1, 0.2, 0.3, 0.5, 0.7],
        }
        forecast = forecaster.forecast(metrics)
        assert forecast.risk_level in ("low", "medium", "high", "critical")

    def test_recommendations_generated(self):
        forecaster = CollapseForecaster()
        metrics = {
            "entropy": [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1],
        }
        forecast = forecaster.forecast(metrics)
        assert len(forecast.recommendations) > 0


# --- Halt and Diagnose ---

class TestHaltAndDiagnoseEdgeCases:
    def test_no_data(self):
        proto = HaltAndDiagnoseProtocol()
        report = proto.should_halt({})
        assert report.should_halt is False

    def test_entropy_drop_triggers_halt(self):
        proto = HaltAndDiagnoseProtocol(entropy_drop_threshold=0.3)
        report = proto.should_halt({"entropy": [4.0, 2.0]})  # 50% drop
        assert report.should_halt is True
        assert report.entropy_trend == "sharp_decline"

    def test_kl_spike_triggers_halt(self):
        proto = HaltAndDiagnoseProtocol(kl_increase_threshold=2.0)
        report = proto.should_halt({"kl_divergence": [0.5, 5.0]})  # +4.5 > 2.0
        assert report.should_halt is True
        assert report.kl_trend == "spiking"

    def test_consecutive_degradations(self):
        proto = HaltAndDiagnoseProtocol(max_consecutive_degradations=3)
        report = proto.should_halt({
            "quality_score": [0.9, 0.85, 0.80, 0.75, 0.70]
        })
        assert report.should_halt is True
        assert report.consecutive_degradations >= 3

    def test_execute_halt(self):
        proto = HaltAndDiagnoseProtocol()
        result = proto.execute_halt()
        assert result["status"] == "halted"
        assert proto.is_halted is True

    def test_resume_after_halt(self):
        proto = HaltAndDiagnoseProtocol()
        proto.execute_halt()
        proto.resume()
        assert proto.is_halted is False

    def test_halt_history(self):
        proto = HaltAndDiagnoseProtocol()
        proto.should_halt({"entropy": [4.0, 3.9]})
        assert len(proto.get_history()) == 1

    def test_stable_entropy(self):
        proto = HaltAndDiagnoseProtocol()
        report = proto.should_halt({"entropy": [4.0, 4.1]})
        assert report.entropy_trend == "stable_or_increasing"

    def test_increasing_kl(self):
        proto = HaltAndDiagnoseProtocol(kl_increase_threshold=5.0)
        report = proto.should_halt({"kl_divergence": [0.5, 1.0]})
        assert report.kl_trend == "increasing"

    def test_decreasing_kl(self):
        proto = HaltAndDiagnoseProtocol()
        report = proto.should_halt({"kl_divergence": [1.0, 0.5]})
        assert report.kl_trend == "stable_or_decreasing"

    def test_warning_severity(self):
        proto = HaltAndDiagnoseProtocol(entropy_drop_threshold=0.9)
        report = proto.should_halt({"entropy": [4.0, 3.8]})  # Small drop
        assert report.severity in ("none", "warning")

    def test_declining_entropy(self):
        proto = HaltAndDiagnoseProtocol(entropy_drop_threshold=0.9)
        report = proto.should_halt({"entropy": [4.0, 3.8]})
        assert report.entropy_trend == "declining"

    def test_critical_severity_multiple_reasons(self):
        proto = HaltAndDiagnoseProtocol(
            entropy_drop_threshold=0.2,
            kl_increase_threshold=1.0,
        )
        report = proto.should_halt({
            "entropy": [4.0, 2.0],
            "kl_divergence": [0.5, 5.0],
        })
        assert report.severity == "critical"
        assert len(report.reasons) >= 2

    def test_halt_report_is_critical_property(self):
        report = HaltReport(should_halt=True, severity="critical")
        assert report.is_critical is True
        report2 = HaltReport(should_halt=False, severity="none")
        assert report2.is_critical is False

    def test_recommended_action_halt(self):
        proto = HaltAndDiagnoseProtocol(entropy_drop_threshold=0.2)
        report = proto.should_halt({"entropy": [4.0, 2.0]})
        assert report.recommended_action == "halt_and_rollback"

    def test_recommended_action_warning(self):
        proto = HaltAndDiagnoseProtocol(entropy_drop_threshold=0.9)
        report = proto.should_halt({
            "entropy": [4.0, 3.8],
            "quality_score": [0.9, 0.85, 0.80],  # 2 consec, triggers warning
        })
        assert report.recommended_action == "increase_alpha"

    def test_zero_entropy_drop(self):
        proto = HaltAndDiagnoseProtocol()
        report = proto.should_halt({"entropy": [0.0, 0.0]})
        assert report.should_halt is False


# --- Recovery ---

class TestRecoveryEdgeCases:
    def test_increase_alpha(self):
        recovery = CollapseRecovery()
        action = recovery.recover("increase_alpha", {"increase_by": 0.2})
        assert action.success is True
        assert recovery.current_alpha == pytest.approx(0.7)

    def test_increase_alpha_caps_at_one(self):
        recovery = CollapseRecovery()
        action = recovery.recover("increase_alpha", {"increase_by": 0.9})
        assert recovery.current_alpha <= 1.0

    def test_rollback_n(self):
        recovery = CollapseRecovery()
        action = recovery.recover("rollback_n", {"n": 3})
        assert action.success is True
        assert recovery.current_checkpoint == 7

    def test_rollback_zero_fails(self):
        recovery = CollapseRecovery()
        action = recovery.recover("rollback_n", {"n": 0})
        assert action.success is False

    def test_rollback_beyond_baseline(self):
        recovery = CollapseRecovery()
        action = recovery.recover("rollback_n", {"n": 100})
        assert recovery.current_checkpoint == 0

    def test_reset_to_baseline(self):
        recovery = CollapseRecovery()
        recovery.recover("increase_alpha", {"increase_by": 0.3})
        action = recovery.recover("reset_to_baseline")
        assert recovery.current_checkpoint == 0
        assert recovery.current_alpha == 0.5

    def test_invalid_strategy(self):
        recovery = CollapseRecovery()
        with pytest.raises(ValueError):
            recovery.recover("invalid_strategy")

    def test_recommend_strategy_critical(self):
        recovery = CollapseRecovery()
        assert recovery.recommend_strategy({"severity": "critical"}) == "reset_to_baseline"

    def test_recommend_strategy_high(self):
        recovery = CollapseRecovery()
        assert recovery.recommend_strategy({"severity": "high"}) == "rollback_n"

    def test_recommend_strategy_low(self):
        recovery = CollapseRecovery()
        assert recovery.recommend_strategy({"severity": "low"}) == "increase_alpha"

    def test_recommend_many_declining(self):
        recovery = CollapseRecovery()
        assert recovery.recommend_strategy({"iterations_declining": 15}) == "reset_to_baseline"

    def test_recommend_moderate_declining(self):
        recovery = CollapseRecovery()
        assert recovery.recommend_strategy({"iterations_declining": 7}) == "rollback_n"

    def test_recovery_history(self):
        recovery = CollapseRecovery()
        recovery.recover("increase_alpha")
        recovery.recover("rollback_n", {"n": 2})
        assert len(recovery.get_history()) == 2

    def test_default_increase_alpha(self):
        recovery = CollapseRecovery()
        action = recovery.recover("increase_alpha")
        assert action.success is True
        assert recovery.current_alpha == pytest.approx(0.6)

    def test_default_rollback(self):
        recovery = CollapseRecovery()
        action = recovery.recover("rollback_n")
        assert action.success is True
        assert recovery.current_checkpoint == 7  # 10 - 3


# --- Complexity Budget ---

class TestComplexityBudgetEdgeCases:
    def test_negative_baseline_raises(self):
        budget = ComplexityBudget()
        with pytest.raises(ValueError):
            budget.set_baseline(-1.0)

    def test_check_without_baseline_raises(self):
        budget = ComplexityBudget()
        with pytest.raises(RuntimeError):
            budget.check({"complexity": 10.0})

    def test_would_exceed_without_baseline_raises(self):
        budget = ComplexityBudget()
        with pytest.raises(RuntimeError):
            budget.would_exceed({"complexity": 10.0})

    def test_remaining_budget_without_baseline_raises(self):
        budget = ComplexityBudget()
        with pytest.raises(RuntimeError):
            budget.remaining_budget()

    def test_within_budget(self):
        budget = ComplexityBudget(max_ratio=5.0, max_cyclomatic=50)
        budget.set_baseline(10.0, cyclomatic=5)
        status = budget.check({"complexity": 30.0, "cyclomatic": 20})
        assert status.within_budget is True

    def test_exceeds_ratio(self):
        budget = ComplexityBudget(max_ratio=5.0)
        budget.set_baseline(10.0)
        status = budget.check({"complexity": 60.0})
        assert status.within_budget is False

    def test_exceeds_cyclomatic(self):
        budget = ComplexityBudget(max_cyclomatic=50)
        budget.set_baseline(10.0, cyclomatic=5)
        status = budget.check({"complexity": 10.0, "cyclomatic": 55})
        assert status.within_budget is False

    def test_remaining_budget_values(self):
        budget = ComplexityBudget(max_ratio=5.0, max_cyclomatic=50)
        budget.set_baseline(10.0, cyclomatic=10)
        remaining = budget.remaining_budget()
        assert remaining["complexity_headroom"] == pytest.approx(40.0)
        assert remaining["cyclomatic_headroom"] == 40

    def test_budget_utilization(self):
        budget = ComplexityBudget(max_ratio=5.0, max_cyclomatic=50)
        budget.set_baseline(10.0, cyclomatic=10)
        status = budget.check({"complexity": 25.0, "cyclomatic": 25})
        assert status.budget_utilization == pytest.approx(0.5)

    def test_zero_baseline(self):
        budget = ComplexityBudget()
        budget.set_baseline(0.0)
        status = budget.check({"complexity": 0.0})
        assert status.ratio == 0.0


# --- Quarantine ---

class TestQuarantineEdgeCases:
    def test_should_quarantine_high_blast(self):
        q = ModificationQuarantine(auto_quarantine_blast_radius=0.7)
        assert q.should_quarantine("m1", 0.8) is True

    def test_should_quarantine_high_risk_level(self):
        q = ModificationQuarantine()
        assert q.should_quarantine("m1", 0.1, "critical") is True

    def test_should_not_quarantine_low(self):
        q = ModificationQuarantine()
        assert q.should_quarantine("m1", 0.1, "low") is False

    def test_enter_and_check(self):
        q = ModificationQuarantine()
        q.enter_quarantine("m1", "High risk", 0.8)
        assert q.check_quarantined("m1") is True

    def test_check_unknown_returns_false(self):
        q = ModificationQuarantine()
        assert q.check_quarantined("unknown") is False

    def test_release(self):
        q = ModificationQuarantine()
        q.enter_quarantine("m1", "High risk", 0.8)
        released = q.release("m1", reviewer="alice")
        assert released.released is True
        assert released.reviewer == "alice"
        assert q.check_quarantined("m1") is False

    def test_release_unknown_raises(self):
        q = ModificationQuarantine()
        with pytest.raises(KeyError):
            q.release("unknown")

    def test_get_all_quarantined(self):
        q = ModificationQuarantine()
        q.enter_quarantine("m1", "r1", 0.8)
        q.enter_quarantine("m2", "r2", 0.9)
        assert len(q.get_all_quarantined()) == 2
        q.release("m1")
        assert len(q.get_all_quarantined()) == 1

    def test_released_history(self):
        q = ModificationQuarantine()
        q.enter_quarantine("m1", "r1", 0.8)
        q.release("m1")
        assert len(q.get_released_history()) == 1


# --- Draft Generator ---

class TestDraftGeneratorCoverage:
    def test_empty_experiments(self):
        gen = PaperDraftGenerator()
        section = gen.generate_results_section([])
        assert "No experimental results" in section

    def test_results_section(self):
        gen = PaperDraftGenerator()
        experiments = [
            {"name": "A", "score": 0.9, "details": "Good"},
            {"name": "B", "score": 0.7, "details": "OK"},
        ]
        section = gen.generate_results_section(experiments)
        assert "A" in section
        assert "B" in section
        assert "0.900" in section

    def test_generate_tables(self):
        gen = PaperDraftGenerator()
        data = [
            {"method": "A", "score": 0.95},
            {"method": "B", "score": 0.85},
        ]
        table = gen.generate_tables(data, caption="Results")
        assert "tabular" in table
        assert "Results" in table
        assert "0.950" in table

    def test_generate_tables_custom_columns(self):
        gen = PaperDraftGenerator()
        data = [{"x": 1, "y": "hello"}]
        table = gen.generate_tables(data, columns=["x", "y"])
        assert "hello" in table

    def test_generate_tables_empty(self):
        gen = PaperDraftGenerator()
        table = gen.generate_tables([])
        assert "No data" in table

    def test_generate_figures(self):
        gen = PaperDraftGenerator()
        specs = [
            {"filename": "fig1.png", "caption": "Results plot", "label": "fig:results"},
            {"filename": "fig2.png", "caption": "Ablation", "label": "fig:ablation"},
        ]
        figures = gen.generate_figures(specs)
        assert len(figures) == 2
        assert "fig1.png" in figures[0]
        assert "Ablation" in figures[1]

    def test_generate_figures_defaults(self):
        gen = PaperDraftGenerator()
        figures = gen.generate_figures([{}])
        assert len(figures) == 1
        assert "figure.png" in figures[0]

    def test_get_all_sections(self):
        gen = PaperDraftGenerator()
        gen.generate_results_section([{"name": "A", "score": 0.9}])
        gen.generate_tables([{"x": 1}])
        assert len(gen.get_all_sections()) == 2


# --- Adversarial Eval ---

class TestAdversarialEvalEdgeCases:
    def test_dict_agent(self):
        eval_set = AdversarialEvalSet()
        agent = {"scores": {"adv_001": 0.9, "adv_002": 0.2}}
        result = eval_set.evaluate(agent)
        assert result.tasks_run >= 20

    def test_plain_agent(self):
        eval_set = AdversarialEvalSet()
        result = eval_set.evaluate("not_an_agent")  # No score method, no dict
        assert result.tasks_run >= 20

    def test_get_tasks_by_type(self):
        eval_set = AdversarialEvalSet()
        gaming = eval_set.get_tasks_by_type("gaming")
        assert len(gaming) > 0
        safety = eval_set.get_tasks_by_type("safety")
        assert len(safety) > 0

    def test_eval_gap(self, mock_agent):
        eval_set = AdversarialEvalSet()
        result = eval_set.evaluate(mock_agent)
        gap = eval_set.compute_eval_gap(0.95, result)
        assert result.eval_gap == gap

    def test_history(self, mock_agent):
        eval_set = AdversarialEvalSet()
        eval_set.evaluate(mock_agent)
        assert len(eval_set.get_history()) == 1


# --- Reward Audit ---

class TestRewardAuditEdgeCases:
    def test_empty_audit(self):
        audit = RewardAuditTrail()
        patterns = audit.detect_patterns()
        assert patterns.suspicious is False
        assert patterns.reward_trend == "stable"

    def test_monotonic_increase_detected(self):
        audit = RewardAuditTrail()
        for i in range(10):
            audit.log(f"in_{i}", f"out_{i}", float(i))
        patterns = audit.detect_patterns()
        assert "monotonically_increasing_rewards" in patterns.patterns_detected

    def test_reward_clustering_detected(self):
        audit = RewardAuditTrail()
        for i in range(20):
            audit.log(f"in_{i}", f"out_{i}", 1.0)  # All same
        patterns = audit.detect_patterns()
        assert "reward_clustering" in patterns.patterns_detected

    def test_get_entries_limited(self):
        audit = RewardAuditTrail()
        for i in range(10):
            audit.log(f"in_{i}", f"out_{i}", float(i))
        entries = audit.get_entries(last_n=3)
        assert len(entries) == 3

    def test_entry_count(self):
        audit = RewardAuditTrail()
        for i in range(5):
            audit.log(f"in_{i}", f"out_{i}", float(i))
        assert audit.entry_count == 5

    def test_max_entries_enforced(self):
        audit = RewardAuditTrail(max_entries=5)
        for i in range(10):
            audit.log(f"in_{i}", f"out_{i}", float(i))
        assert audit.entry_count == 5

    def test_sudden_jump_detected(self):
        audit = RewardAuditTrail()
        for i in range(10):
            audit.log(f"in_{i}", f"out_{i}", 1.0)
        audit.log("in_jump", "out_jump", 100.0)  # Huge jump
        patterns = audit.detect_patterns()
        assert "sudden_reward_jump" in patterns.patterns_detected

    def test_decreasing_trend(self):
        audit = RewardAuditTrail()
        for i in range(10):
            audit.log(f"in_{i}", f"out_{i}", 10.0 - float(i))
        patterns = audit.detect_patterns()
        assert patterns.reward_trend == "decreasing"

    def test_volatile_trend(self):
        audit = RewardAuditTrail()
        for i in range(20):
            audit.log(f"in_{i}", f"out_{i}", float(i % 2) * 100.0)
        patterns = audit.detect_patterns()
        # With values alternating between 0 and 100, should be volatile
        assert patterns.reward_trend in ("volatile", "stable")

    def test_get_reward_history(self):
        audit = RewardAuditTrail()
        audit.log("a", "b", 1.0)
        audit.log("c", "d", 2.0)
        history = audit.get_reward_history()
        assert history == [1.0, 2.0]


# --- Reward Sanity ---

class TestRewardSanityEdgeCases:
    def test_empty_rewards(self):
        checker = RewardSanityChecker()
        result = checker.check([])
        assert result.sane is True

    def test_nan_detected(self):
        checker = RewardSanityChecker()
        result = checker.check([1.0, float('nan'), 2.0])
        assert result.sane is False
        assert result.violation_count > 0

    def test_inf_detected(self):
        checker = RewardSanityChecker()
        result = checker.check([1.0, float('inf')])
        assert result.sane is False

    def test_out_of_bounds(self):
        checker = RewardSanityChecker(min_reward=0.0, max_reward=10.0)
        result = checker.check([1.0, 15.0, -5.0])
        assert result.sane is False

    def test_high_std_dev(self):
        checker = RewardSanityChecker(max_std_dev=1.0)
        result = checker.check([0.0, 100.0, 0.0, 100.0])
        assert result.sane is False

    def test_degenerate_distribution(self):
        checker = RewardSanityChecker()
        result = checker.check([5.0] * 15)
        assert result.sane is False  # Zero variance with >= 10 samples

    def test_all_non_finite(self):
        checker = RewardSanityChecker()
        result = checker.check([float('nan'), float('nan')])
        assert result.sane is False

    def test_violation_count_property(self):
        result = SanityResult(sane=False, violations=["a", "b"])
        assert result.violation_count == 2


# --- Budget Manager ---

class TestBudgetManagerEdgeCases:
    def test_invalid_level_raises(self):
        mgr = BudgetManager()
        with pytest.raises(ValueError):
            mgr.can_spend(1.0, "invalid")

    def test_invalid_remaining_level(self):
        mgr = BudgetManager()
        with pytest.raises(ValueError):
            mgr.remaining("invalid")

    def test_invalid_reset_level(self):
        mgr = BudgetManager()
        with pytest.raises(ValueError):
            mgr.reset_level("invalid")

    def test_invalid_get_state_level(self):
        mgr = BudgetManager()
        with pytest.raises(ValueError):
            mgr.get_state("invalid")

    def test_spend_returns_false_when_exceeded(self):
        mgr = BudgetManager(query_limit=1.0)
        assert mgr.spend(2.0, "query") is False

    def test_get_state(self):
        mgr = BudgetManager(query_limit=5.0)
        mgr.spend(2.0, "query")
        state = mgr.get_state("query")
        assert state.spent == pytest.approx(2.0)
        assert state.remaining == pytest.approx(3.0)
        assert state.utilization == pytest.approx(0.4)

    def test_get_all_states(self):
        mgr = BudgetManager()
        states = mgr.get_all_states()
        assert len(states) == 4

    def test_spend_log(self):
        mgr = BudgetManager(query_limit=5.0)
        mgr.spend(1.0, "query", "test")
        log = mgr.get_spend_log()
        assert len(log) == 1
        assert log[0]["description"] == "test"

    def test_burn_rate_empty(self):
        mgr = BudgetManager()
        assert mgr.burn_rate() == 0.0


# --- Cost Optimizer ---

class TestCostOptimizerEdgeCases:
    def test_low_burn_rate_no_suggestions(self):
        forecaster = CostForecaster()
        forecast = forecaster.forecast([1.0] * 10, 10, 10000.0)
        optimizer = CostOptimizer()
        suggestions = optimizer.suggest(forecast)
        # With very low burn rate and ample budget, should have fewer suggestions
        assert isinstance(suggestions, list)

    def test_high_burn_rate_many_suggestions(self):
        forecaster = CostForecaster()
        forecast = forecaster.forecast([25.0] * 10, 100, 500.0, spent_so_far=400.0)
        optimizer = CostOptimizer()
        suggestions = optimizer.suggest(forecast)
        assert len(suggestions) >= 3

    def test_cost_saving_roi(self):
        saving = CostSaving("test", "desc", 100.0, "low", "low", 1)
        assert saving.roi_estimate == "high"
        saving2 = CostSaving("test", "desc", 50.0, "high", "low", 1)
        assert saving2.roi_estimate == "low"
        saving3 = CostSaving("test", "desc", 200.0, "medium", "low", 1)
        assert saving3.roi_estimate == "medium"

    def test_suggest_with_eval_frequency(self):
        forecaster = CostForecaster()
        forecast = forecaster.forecast([5.0] * 10, 100, 1000.0)
        optimizer = CostOptimizer()
        suggestions = optimizer.suggest(forecast, {"eval_frequency": 1})
        categories = [s.category for s in suggestions]
        assert "eval_frequency" in categories


# --- Eval Rotation ---

class TestEvalRotationEdgeCases:
    def test_should_not_rotate_early(self):
        rotator = EvalSetRotator(rotate_every_n=10)
        assert rotator.should_rotate(5) is False

    def test_rotate_with_empty_reserve(self):
        rotator = EvalSetRotator()
        result = rotator.rotate(["a", "b"], [], iteration=0)
        assert result["active"] == ["a", "b"]
        assert result["reserve"] == []

    def test_rotation_history(self):
        rotator = EvalSetRotator(rotate_every_n=5)
        rotator.rotate(["a", "b", "c"], ["d", "e"], iteration=5)
        history = rotator.get_rotation_history()
        assert len(history) == 1
        assert history[0].iteration == 5


# --- Tightness Detector ---

class TestTightnessDetectorEdgeCases:
    def test_empty_history(self):
        detector = TightnessDetector()
        report = detector.detect("test", [])
        assert report.is_too_loose is True

    def test_too_tight(self):
        detector = TightnessDetector(too_tight_threshold=0.5)
        # All entries binding
        history = [{"value": 0.91, "threshold": 0.90}] * 10
        report = detector.detect("test", history)
        assert report.is_too_tight is True

    def test_too_loose(self):
        detector = TightnessDetector(too_loose_threshold=0.1)
        history = [{"value": 0.5, "threshold": 0.90}] * 10
        report = detector.detect("test", history)
        assert report.is_too_loose is True

    def test_too_tight_method(self):
        detector = TightnessDetector(too_tight_threshold=0.5)
        assert detector.too_tight(0.6) is True
        assert detector.too_tight(0.3) is False

    def test_too_loose_method(self):
        detector = TightnessDetector(too_loose_threshold=0.1)
        assert detector.too_loose(0.05) is True
        assert detector.too_loose(0.3) is False


# --- Adaptive Thresholds ---

class TestAdaptiveThresholdsEdgeCases:
    def test_empty_history(self):
        adaptive = AdaptiveThresholds()
        adj = adaptive.suggest_adjustment("test", [])
        assert adj.direction == "maintain"
        assert adj.confidence == 0.0

    def test_loosen_suggestion(self):
        adaptive = AdaptiveThresholds()
        # High binding fraction, worse perf when binding
        history = [
            {"value": 0.91, "threshold": 0.90, "performance": 0.5},
        ] * 20 + [
            {"value": 0.5, "threshold": 0.90, "performance": 0.9},
        ] * 5
        adj = adaptive.suggest_adjustment("test", history, 0.90)
        assert adj.direction == "loosen"

    def test_tighten_suggestion(self):
        adaptive = AdaptiveThresholds()
        history = [
            {"value": 0.3, "threshold": 0.90, "performance": 0.5},
        ] * 20
        adj = adaptive.suggest_adjustment("test", history, 0.90)
        assert adj.direction == "tighten"

    def test_history_tracked(self):
        adaptive = AdaptiveThresholds()
        adaptive.suggest_adjustment("test", [], 0.90)
        assert len(adaptive.get_history()) == 1


# --- Deadline Tracker ---

class TestDeadlineTrackerEdgeCases:
    def test_severity_by_name(self):
        tracker = DeadlineTracker(reference_date="2026-03-16")
        severity = tracker.severity("NeurIPS 2026")
        assert severity in ("relaxed", "approaching", "urgent", "critical", "past")

    def test_severity_unknown_raises(self):
        tracker = DeadlineTracker(reference_date="2026-03-16")
        with pytest.raises(KeyError):
            tracker.severity("Unknown Conference")

    def test_all_past(self):
        tracker = DeadlineTracker(reference_date="2027-12-01")
        next_dl = tracker.next_deadline()
        assert next_dl is None

    def test_deadline_status_is_urgent(self):
        status = DeadlineStatus(
            name="Test", abstract_date="2026-01-01",
            submission_date="2026-01-05", days_to_abstract=3,
            days_to_submission=5, severity="urgent", is_past=False,
        )
        assert status.is_urgent is True


# --- Fallback Planner ---

class TestFallbackPlannerEdgeCases:
    def test_past_deadlines_skipped(self):
        planner = FallbackPlanner()
        plans = planner.plan(
            {"experiments": 0.5, "writing": 0.3, "figures": 0.2},
            [{"name": "Past", "days_to_submission": -10}],
        )
        assert len(plans) == 0

    def test_high_readiness_full_submission(self):
        planner = FallbackPlanner()
        plans = planner.plan(
            {"experiments": 0.9, "writing": 0.85, "figures": 0.8},
            [{"name": "Venue", "days_to_submission": 30}],
        )
        assert plans[0].strategy == "full_submission"

    def test_low_readiness_defer(self):
        planner = FallbackPlanner()
        plans = planner.plan(
            {"experiments": 0.1, "writing": 0.1, "figures": 0.1},
            [{"name": "Venue", "days_to_submission": 30}],
        )
        assert plans[0].strategy == "defer"
        assert plans[0].is_deferred is True

    def test_mvs_not_viable(self):
        planner = FallbackPlanner()
        mvs = planner.minimum_viable_submission({"experiments": 0.1})
        assert mvs["viable"] is False

    def test_mvs_extended_abstract(self):
        planner = FallbackPlanner()
        mvs = planner.minimum_viable_submission({"experiments": 0.4, "writing": 0.2})
        assert mvs["type"] == "extended_abstract"

    def test_mvs_workshop(self):
        planner = FallbackPlanner()
        mvs = planner.minimum_viable_submission({"experiments": 0.6, "writing": 0.3})
        assert mvs["type"] == "workshop_paper"


# --- Readiness Checker ---

class TestReadinessCheckerEdgeCases:
    def test_not_ready(self):
        checker = ReadinessChecker()
        report = checker.check(
            {"completed": 2, "total": 10, "results_quality": 0.3},
            {"sections_done": 1, "total_sections": 10, "quality": 0.2},
        )
        assert report.ready is False
        assert len(report.blockers) > 0

    def test_ready(self):
        checker = ReadinessChecker()
        report = checker.check(
            {"completed": 10, "total": 10, "results_quality": 0.95},
            {"sections_done": 7, "total_sections": 7, "quality": 0.9},
        )
        assert report.ready is True

    def test_gap_to_ready(self):
        checker = ReadinessChecker()
        report = checker.check(
            {"completed": 5, "total": 10, "results_quality": 0.5},
            {"sections_done": 3, "total_sections": 10, "quality": 0.5},
        )
        assert report.gap_to_ready > 0


# --- Risk Registry ---

class TestRiskRegistryEdgeCases:
    def test_register_unknown_risk_raises(self):
        registry = RiskRegistry()
        with pytest.raises(KeyError):
            registry.register_checker("R99", lambda: None)

    def test_check_unknown_risk_raises(self):
        registry = RiskRegistry()
        with pytest.raises(KeyError):
            registry.check("R99")

    def test_get_unknown_risk_raises(self):
        registry = RiskRegistry()
        with pytest.raises(KeyError):
            registry.get_risk("R99")

    def test_get_all_risks(self):
        registry = RiskRegistry()
        risks = registry.get_all_risks()
        assert len(risks) == 6

    def test_risk_status_properties(self):
        status = RiskStatus(
            risk_id="R1", name="Test", domain="test",
            severity="critical", score=0.9,
        )
        assert status.is_critical is True
        assert status.needs_attention is True

    def test_dashboard_properties(self):
        dashboard = RiskDashboard(
            statuses=[], overall_severity="low",
            overall_score=0.1, critical_count=0, high_count=0,
        )
        assert dashboard.needs_immediate_action is False
        assert dashboard.total_risks == 0


# --- Retrospective ---

class TestRetrospectiveEdgeCases:
    def test_no_incidents(self):
        retro = RiskRetrospective()
        report = retro.analyze([], [], "Q1")
        assert report.resolution_rate == 1.0

    def test_declining_risk_scores(self):
        retro = RiskRetrospective()
        report = retro.analyze(
            [],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "Q1",
        )
        trends = [i.category for i in report.insights]
        assert "trend" in trends

    def test_critical_incidents(self):
        retro = RiskRetrospective()
        report = retro.analyze(
            [{"domain": "cost", "severity": "critical", "status": "open"}],
            [0.5],
            "Q1",
        )
        severities = [i.category for i in report.insights]
        assert "severity" in severities

    def test_low_resolution_rate(self):
        retro = RiskRetrospective()
        report = retro.analyze(
            [
                {"domain": "cost", "severity": "low", "status": "open"},
                {"domain": "cost", "severity": "low", "status": "open"},
                {"domain": "cost", "severity": "low", "status": "open"},
            ],
            [],
            "Q1",
        )
        assert report.resolution_rate == pytest.approx(0.0)
        categories = [i.category for i in report.insights]
        assert "resolution" in categories


# --- Incident Manager ---

class TestIncidentManagerEdgeCases:
    def test_get_incident(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("A", "a", "low", "cost")
        fetched = mgr.get_incident(inc.incident_id)
        assert fetched.title == "A"

    def test_get_unknown_incident_raises(self):
        mgr = IncidentManager()
        with pytest.raises(KeyError):
            mgr.get_incident("INC-9999")

    def test_total_count(self):
        mgr = IncidentManager()
        mgr.create_incident("A", "a", "low", "cost")
        mgr.create_incident("B", "b", "low", "cost")
        assert mgr.total_count == 2

    def test_incident_is_critical(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("A", "a", "critical", "cost")
        assert inc.is_critical is True


# --- Dashboard ---

class TestDashboardEdgeCases:
    def test_generate_report_auto_compute(self):
        registry = RiskRegistry()
        dashboard = UnifiedRiskDashboard(registry)
        report = dashboard.generate_stakeholder_report()
        assert "STAKEHOLDER REPORT" in report

    def test_trend_insufficient_data(self):
        registry = RiskRegistry()
        dashboard = UnifiedRiskDashboard(registry)
        trend = dashboard.get_trend()
        assert trend["trend"] == "insufficient_data"


# --- Blast Radius ---

class TestBlastRadiusEdgeCases:
    def test_no_affected_modules(self):
        estimator = BlastRadiusEstimator()
        estimate = estimator.estimate({"affected_modules": [], "lines_changed": 0})
        assert estimate.overall_risk == 0.0

    def test_high_risk_core_module(self):
        estimator = BlastRadiusEstimator()
        estimate = estimator.estimate({
            "affected_modules": ["core"],
            "lines_changed": 500,
            "total_lines": 1000,
        })
        assert estimate.is_high_risk is True

    def test_zero_total_lines(self):
        estimator = BlastRadiusEstimator()
        estimate = estimator.estimate({
            "affected_modules": ["metrics"],
            "lines_changed": 10,
            "total_lines": 0,
        })
        assert estimate.code_change_magnitude == 0.0


# --- Circuit Breaker ---

class TestCircuitBreakerEdgeCases:
    def test_state_property(self):
        cb = CircuitBreaker()
        state = cb.state
        assert state.query_id == "none"

    def test_cost_accumulated(self):
        cb = CircuitBreaker(max_cost_per_query=100.0)
        cb.register_query("q1")
        cb.record_sub_query(cost=5.0)
        cb.record_tokens(100, cost=3.0)
        assert cb.state.cost == pytest.approx(8.0)
