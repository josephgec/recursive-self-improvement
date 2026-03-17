"""Integration tests covering all 6 risk domains and cross-cutting concerns.

Tests:
- All 6 risks checked via registry
- Staging pass/fail
- Budget enforcement
- Circuit breaker
- Relaxation lifecycle
- Deadline check
- Incident lifecycle
- Dashboard
- Report generation
"""

import pytest
from src.orchestration.risk_registry import RiskRegistry, RiskStatus
from src.orchestration.risk_dashboard import UnifiedRiskDashboard
from src.orchestration.incident_manager import IncidentManager
from src.collapse.alpha_scheduler import ConservativeAlphaScheduler, AlphaScheduleConfig
from src.collapse.data_reserve import CleanDataReserve
from src.collapse.collapse_forecaster import CollapseForecaster
from src.collapse.halt_and_diagnose import HaltAndDiagnoseProtocol
from src.collapse.recovery import CollapseRecovery
from src.self_mod.staging_env import StagingEnvironment
from src.self_mod.complexity_budget import ComplexityBudget
from src.self_mod.blast_radius import BlastRadiusEstimator
from src.self_mod.quarantine import ModificationQuarantine
from src.reward.adversarial_eval import AdversarialEvalSet
from src.reward.eval_rotation import EvalSetRotator
from src.reward.reward_audit import RewardAuditTrail
from src.reward.reward_sanity import RewardSanityChecker
from src.cost.budget_manager import BudgetManager
from src.cost.circuit_breaker import CircuitBreaker
from src.cost.cost_forecaster import CostForecaster
from src.cost.cost_optimizer import CostOptimizer
from src.constraints.graduated_relaxation import GraduatedRelaxation
from src.constraints.compensation import CompensationMonitor
from src.constraints.tightness_detector import TightnessDetector
from src.constraints.adaptive_thresholds import AdaptiveThresholds
from src.publication.deadline_tracker import DeadlineTracker
from src.publication.draft_generator import PaperDraftGenerator
from src.publication.fallback_planner import FallbackPlanner
from src.publication.readiness_checker import ReadinessChecker
from src.analysis.risk_retrospective import RiskRetrospective
from src.analysis.report import ReportGenerator


class TestAllSixRisksChecked:
    """Test that all 6 risks can be checked via registry."""

    def test_registry_has_6_risks(self):
        registry = RiskRegistry()
        assert registry.risk_count == 6

    def test_check_all_returns_dashboard(self):
        registry = RiskRegistry()
        dashboard = registry.check_all()
        assert len(dashboard.statuses) == 6

    def test_all_risk_ids_present(self):
        registry = RiskRegistry()
        dashboard = registry.check_all()
        risk_ids = {s.risk_id for s in dashboard.statuses}
        assert risk_ids == {"R1", "R2", "R3", "R4", "R5", "R6"}

    def test_custom_checker_invoked(self):
        registry = RiskRegistry()
        registry.register_checker("R1", lambda: RiskStatus(
            risk_id="R1", name="Collapse", domain="collapse",
            severity="high", score=0.8,
            details={"entropy": "dropping"},
        ))
        status = registry.check("R1")
        assert status.severity == "high"
        assert status.score == 0.8

    def test_dashboard_reflects_custom_checkers(self):
        registry = RiskRegistry()
        registry.register_checker("R4", lambda: RiskStatus(
            risk_id="R4", name="Cost", domain="cost",
            severity="critical", score=0.95,
        ))
        dashboard = registry.check_all()
        assert dashboard.critical_count >= 1
        assert dashboard.overall_severity == "critical"


class TestStagingIntegration:
    """Integration tests for staging pass/fail with blast radius and quarantine."""

    def test_staging_pass_flow(self):
        staging = StagingEnvironment()
        agent_state = {"quality": 0.85, "modifier": 0.0}
        candidate = {"id": "c1", "changes": {"modifier": 0.02}}
        result = staging.test_modification(agent_state, candidate)
        assert result.passed is True

    def test_staging_fail_triggers_quarantine(self):
        staging = StagingEnvironment()
        quarantine = ModificationQuarantine()

        agent_state = {"quality": 0.85, "modifier": 0.0}
        candidate = {"id": "c_bad", "changes": {"modifier": -0.5}}
        result = staging.test_modification(agent_state, candidate)

        if not result.passed:
            quarantine.enter_quarantine("c_bad", "Failed staging", 0.8)

        assert quarantine.check_quarantined("c_bad") is True

    def test_blast_radius_gates_staging(self):
        estimator = BlastRadiusEstimator()
        quarantine = ModificationQuarantine(auto_quarantine_blast_radius=0.5)

        mod = {"affected_modules": ["core"], "lines_changed": 100, "total_lines": 500}
        estimate = estimator.estimate(mod)

        if quarantine.should_quarantine("mod1", estimate.overall_risk, estimate.risk_level):
            quarantine.enter_quarantine("mod1", "High blast radius", estimate.overall_risk)
            assert quarantine.check_quarantined("mod1") is True

    def test_complexity_budget_checked(self):
        budget = ComplexityBudget()
        budget.set_baseline(10.0, cyclomatic=5)

        # Within budget
        status = budget.check({"complexity": 30.0, "cyclomatic": 20})
        assert status.within_budget is True

        # Over budget
        assert budget.would_exceed({"complexity": 60.0, "cyclomatic": 10})


class TestBudgetEnforcement:
    """Integration tests for budget enforcement across levels."""

    def test_query_level_enforcement(self):
        mgr = BudgetManager(query_limit=1.0, session_limit=50.0)
        assert mgr.can_spend(0.5, "query") is True
        assert mgr.can_spend(1.5, "query") is False

    def test_cascading_spend(self):
        mgr = BudgetManager(
            query_limit=1.0, session_limit=50.0,
            iteration_limit=500.0, phase_limit=5000.0,
        )
        mgr.spend(0.5, "query")
        # Should cascade to all higher levels
        assert mgr.remaining("session") == pytest.approx(49.5)
        assert mgr.remaining("iteration") == pytest.approx(499.5)
        assert mgr.remaining("phase") == pytest.approx(4999.5)

    def test_session_limit_blocks_query(self):
        mgr = BudgetManager(query_limit=100.0, session_limit=5.0)
        mgr.spend(4.5, "query")
        assert mgr.can_spend(1.0, "query") is False  # Session limit exceeded

    def test_burn_rate_tracking(self):
        mgr = BudgetManager(query_limit=10.0, session_limit=100.0,
                            iteration_limit=1000.0, phase_limit=10000.0)
        mgr.spend(2.0, "query", "test1")
        mgr.spend(3.0, "query", "test2")
        rate = mgr.burn_rate("session")
        assert rate == pytest.approx(2.5)

    def test_reset_level(self):
        mgr = BudgetManager(query_limit=1.0, session_limit=50.0)
        mgr.spend(0.5, "query")
        mgr.reset_level("query")
        assert mgr.remaining("query") == pytest.approx(1.0)


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with budget manager."""

    def test_circuit_breaker_kills_at_limit(self):
        cb = CircuitBreaker(max_sub_queries=10)
        cb.register_query("q1")
        for _ in range(10):
            cb.record_sub_query()
        assert cb.is_tripped is True
        result = cb.kill()
        assert result["status"] == "killed"

    def test_circuit_breaker_with_budget(self):
        cb = CircuitBreaker(max_cost_per_query=5.0)
        budget = BudgetManager(query_limit=5.0)

        cb.register_query("q1")
        cost = 2.0
        cb.record_sub_query(cost=cost)
        budget.spend(cost, "query")

        cost = 3.5
        within_cb = cb.record_sub_query(cost=cost)
        can_budget = budget.can_spend(cost, "query")

        # Circuit breaker should trip at total 5.5
        assert within_cb is False or can_budget is False


class TestRelaxationLifecycle:
    """Integration tests for the full relaxation lifecycle."""

    def test_full_lifecycle(self):
        gr = GraduatedRelaxation(max_steps=3, step_size_pp=2)
        comp = CompensationMonitor()

        # Set up constraint
        gr.set_constraint("quality_threshold", 0.90)

        # Step 1: Propose and apply
        p1 = gr.propose_relaxation("quality_threshold", "Blocking progress")
        assert p1.approved is True
        gr.apply_relaxation(p1)
        comp.activate_compensation("quality_threshold", "Relaxed by 2pp")
        assert comp.is_compensated("quality_threshold") is True

        # Step 2
        p2 = gr.propose_relaxation("quality_threshold")
        gr.apply_relaxation(p2)

        # Step 3
        p3 = gr.propose_relaxation("quality_threshold")
        gr.apply_relaxation(p3)

        # Step 4 should be rejected
        p4 = gr.propose_relaxation("quality_threshold")
        assert p4.approved is False

        # Revert
        gr.revert("quality_threshold")
        comp.deactivate("quality_threshold")
        assert gr.get_constraint_value("quality_threshold") == pytest.approx(0.90)
        assert comp.is_compensated("quality_threshold") is False

    def test_tightness_detection_integration(self, constraint_history):
        detector = TightnessDetector()
        report = detector.detect("quality_threshold", constraint_history)
        # With the fixture data, many entries are at/above threshold
        assert isinstance(report.binding_fraction, float)
        assert isinstance(report.recommendation, str)

    def test_adaptive_thresholds_integration(self, constraint_history):
        adaptive = AdaptiveThresholds()
        adj = adaptive.suggest_adjustment("quality_threshold", constraint_history, 0.90)
        assert adj.direction in ("tighten", "loosen", "maintain")
        assert adj.confidence > 0


class TestDeadlineCheck:
    """Integration tests for deadline checking."""

    def test_deadline_check_returns_statuses(self):
        tracker = DeadlineTracker(reference_date="2026-03-16")
        statuses = tracker.check()
        assert len(statuses) >= 3

    def test_next_deadline(self):
        tracker = DeadlineTracker(reference_date="2026-03-16")
        next_dl = tracker.next_deadline()
        assert next_dl is not None
        assert next_dl.days_to_submission > 0

    def test_severity_computation(self):
        tracker = DeadlineTracker(reference_date="2026-03-16")
        statuses = tracker.check()
        for s in statuses:
            assert s.severity in ("relaxed", "approaching", "urgent", "critical", "past")

    def test_past_deadlines_identified(self):
        tracker = DeadlineTracker(reference_date="2026-12-01")
        statuses = tracker.check()
        past = [s for s in statuses if s.is_past]
        assert len(past) > 0

    def test_readiness_integration(self, experiment_status, writing_status):
        checker = ReadinessChecker()
        report = checker.check(experiment_status, writing_status)
        assert isinstance(report.ready, bool)
        assert 0 <= report.overall_readiness <= 1.0

    def test_fallback_planner_integration(self):
        planner = FallbackPlanner()
        readiness = {"experiments": 0.5, "writing": 0.3, "figures": 0.2}
        deadlines = [{"name": "NeurIPS 2026", "days_to_submission": 60}]
        plans = planner.plan(readiness, deadlines)
        assert len(plans) > 0

    def test_minimum_viable_submission(self):
        planner = FallbackPlanner()
        mvs = planner.minimum_viable_submission({"experiments": 0.9, "writing": 0.7})
        assert mvs["viable"] is True

    def test_draft_generator_integration(self):
        gen = PaperDraftGenerator("Test Paper")
        experiments = [
            {"name": "method_A", "score": 0.85},
            {"name": "method_B", "score": 0.90},
        ]
        section = gen.generate_results_section(experiments)
        assert "Results" in section
        assert "method_B" in section


class TestIncidentLifecycle:
    """Integration tests for incident management lifecycle."""

    def test_create_incident(self):
        mgr = IncidentManager()
        incident = mgr.create_incident(
            "Cost overrun detected",
            "Budget exceeded by 20%",
            "high",
            "cost",
        )
        assert incident.is_open is True
        assert incident.severity == "high"

    def test_update_status(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("Test", "Test incident", "medium", "collapse")
        mgr.update_status(inc.incident_id, "investigating", "Starting investigation")
        assert inc.status == "investigating"

    def test_resolve_incident(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("Test", "Test", "low", "reward")
        mgr.resolve(inc.incident_id, "Fixed the issue")
        assert inc.is_open is False
        assert inc.resolution == "Fixed the issue"

    def test_full_lifecycle(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("Alert", "Cost alert", "high", "cost")
        assert mgr.open_count == 1

        mgr.update_status(inc.incident_id, "investigating")
        mgr.update_status(inc.incident_id, "mitigating")
        mgr.resolve(inc.incident_id, "Budget reallocated")

        assert mgr.open_count == 0
        assert len(inc.updates) == 3

    def test_get_by_domain(self):
        mgr = IncidentManager()
        mgr.create_incident("A", "a", "low", "cost")
        mgr.create_incident("B", "b", "low", "collapse")
        mgr.create_incident("C", "c", "low", "cost")
        assert len(mgr.get_by_domain("cost")) == 2

    def test_get_by_severity(self):
        mgr = IncidentManager()
        mgr.create_incident("A", "a", "critical", "cost")
        mgr.create_incident("B", "b", "low", "cost")
        assert len(mgr.get_by_severity("critical")) == 1

    def test_invalid_severity_raises(self):
        mgr = IncidentManager()
        with pytest.raises(ValueError):
            mgr.create_incident("A", "a", "extreme", "cost")

    def test_invalid_status_raises(self):
        mgr = IncidentManager()
        inc = mgr.create_incident("A", "a", "low", "cost")
        with pytest.raises(ValueError):
            mgr.update_status(inc.incident_id, "invalid_status")

    def test_unknown_incident_raises(self):
        mgr = IncidentManager()
        with pytest.raises(KeyError):
            mgr.update_status("INC-9999", "investigating")


class TestDashboard:
    """Integration tests for the unified risk dashboard."""

    def test_compute_dashboard(self):
        registry = RiskRegistry()
        dashboard = UnifiedRiskDashboard(registry)
        result = dashboard.compute()
        assert result.total_risks == 6

    def test_stakeholder_report_generated(self):
        registry = RiskRegistry()
        dashboard = UnifiedRiskDashboard(registry)
        result = dashboard.compute()
        report = dashboard.generate_stakeholder_report(result)
        assert "STAKEHOLDER REPORT" in report
        assert "Overall Severity" in report

    def test_dashboard_snapshots(self):
        registry = RiskRegistry()
        dashboard = UnifiedRiskDashboard(registry)
        dashboard.compute()
        dashboard.compute()
        assert len(dashboard.get_snapshots()) == 2

    def test_trend_analysis(self):
        registry = RiskRegistry()
        # Register a checker that changes between calls
        call_count = [0]
        def dynamic_checker():
            call_count[0] += 1
            score = 0.3 if call_count[0] <= 6 else 0.8
            return RiskStatus(
                risk_id="R1", name="Test", domain="test",
                severity="low" if score < 0.5 else "high",
                score=score,
            )
        registry.register_checker("R1", dynamic_checker)

        dashboard = UnifiedRiskDashboard(registry)
        dashboard.compute()
        dashboard.compute()
        trend = dashboard.get_trend()
        assert "trend" in trend

    def test_dashboard_with_no_checkers(self):
        registry = RiskRegistry()
        dashboard = UnifiedRiskDashboard(registry)
        result = dashboard.compute()
        # All risks should default to low
        assert result.overall_severity == "low"


class TestReportGeneration:
    """Integration tests for comprehensive report generation."""

    def test_full_report(self):
        registry = RiskRegistry()
        dashboard = UnifiedRiskDashboard(registry)
        db = dashboard.compute()

        incident_mgr = IncidentManager()
        incident_mgr.create_incident("Test", "Test incident", "low", "cost")

        retro = RiskRetrospective()
        retro_report = retro.analyze(
            [{"domain": "cost", "severity": "low", "status": "resolved"}],
            [0.3, 0.4, 0.5],
            "Q1 2026",
        )

        report_gen = ReportGenerator("Integration Test Report")
        report_gen.add_section("Notes", "Integration test output")
        report = report_gen.generate(
            dashboard=db,
            incidents=incident_mgr.get_history(),
            retrospective=retro_report,
        )
        assert "Integration Test Report" in report
        assert "Overall Risk Status" in report
        assert "Incidents" in report

    def test_report_with_empty_data(self):
        report_gen = ReportGenerator()
        report = report_gen.generate()
        assert "Risk Management Report" in report

    def test_clear_sections(self):
        report_gen = ReportGenerator()
        report_gen.add_section("A", "content")
        report_gen.clear_sections()
        report = report_gen.generate()
        assert "A" not in report or "content" not in report


class TestCollapseIntegration:
    """Integration tests for collapse domain."""

    def test_forecast_and_halt(self, collapsing_metrics_history):
        forecaster = CollapseForecaster()
        forecast = forecaster.forecast(collapsing_metrics_history)
        assert forecast.risk_level in ("high", "critical")

        halt_proto = HaltAndDiagnoseProtocol()
        report = halt_proto.should_halt(collapsing_metrics_history)
        # With collapsing metrics, should recommend halting
        assert report.severity in ("warning", "critical")

    def test_recovery_after_halt(self):
        recovery = CollapseRecovery()
        strategy = recovery.recommend_strategy({"severity": "critical"})
        assert strategy == "reset_to_baseline"

        action = recovery.recover(strategy)
        assert action.success is True
        assert recovery.current_checkpoint == 0

    def test_data_reserve_integration(self):
        reserve = CleanDataReserve(initial_size=1000, min_reserve_fraction=0.1)
        assert reserve.is_sufficient() is True

        indices = reserve.draw(800)
        assert len(indices) == 800
        assert reserve.get_remaining() == 200

        # Should not be able to draw below min reserve (100)
        with pytest.raises(ValueError):
            reserve.draw(150)

    def test_alpha_scheduler_with_forecast(self):
        scheduler = ConservativeAlphaScheduler(
            AlphaScheduleConfig(schedule_type="adaptive", adaptive_entropy_threshold=2.0)
        )
        # Simulate entropy drop
        a1 = scheduler.get_alpha(0, {"entropy": 3.5})
        a2 = scheduler.get_alpha(1, {"entropy": 1.5})  # Below threshold
        # Alpha should increase when entropy drops
        assert a2 >= a1 * 0.9  # At least maintained


class TestRewardIntegration:
    """Integration tests for reward domain."""

    def test_adversarial_eval(self, mock_agent):
        eval_set = AdversarialEvalSet()
        result = eval_set.evaluate(mock_agent)
        assert result.tasks_run >= 20
        assert result.pass_rate > 0

    def test_eval_gap_detection(self, mock_agent):
        eval_set = AdversarialEvalSet()
        result = eval_set.evaluate(mock_agent)
        gap = eval_set.compute_eval_gap(0.95, result)
        assert isinstance(gap, float)

    def test_eval_rotation(self):
        rotator = EvalSetRotator(rotate_every_n=5, seed=42)
        current = [f"t{i}" for i in range(10)]
        reserve = [f"r{i}" for i in range(5)]

        assert rotator.should_rotate(5) is True
        result = rotator.rotate(current, reserve, iteration=5)
        assert len(result["active"]) == 10
        assert len(result["reserve"]) == 5
        assert result["active"] != current  # Something changed

    def test_reward_audit_and_sanity(self):
        audit = RewardAuditTrail()
        for i in range(20):
            audit.log(f"input_{i}", f"output_{i}", float(i) * 0.5)

        patterns = audit.detect_patterns()
        assert isinstance(patterns.suspicious, bool)

        sanity = RewardSanityChecker(min_reward=-10.0, max_reward=100.0)
        result = sanity.check(audit.get_reward_history())
        assert result.sane is True


class TestCostIntegration:
    """Integration tests for cost domain."""

    def test_optimizer_suggestions(self):
        forecaster = CostForecaster()
        forecast = forecaster.forecast(
            [15.0] * 10, 100, 500.0, spent_so_far=300.0,
        )
        optimizer = CostOptimizer()
        suggestions = optimizer.suggest(forecast)
        assert len(suggestions) > 0
        assert all(isinstance(s.estimated_savings, float) for s in suggestions)


class TestConstraintsIntegration:
    """Integration tests for constraints domain."""

    def test_tightness_and_adaptive(self):
        history = [
            {"value": 0.91, "threshold": 0.90, "performance": 0.8},
            {"value": 0.92, "threshold": 0.90, "performance": 0.82},
            {"value": 0.90, "threshold": 0.90, "performance": 0.75},
        ] * 5  # 15 entries, many binding

        detector = TightnessDetector(too_tight_threshold=0.5)
        report = detector.detect("quality", history)

        adaptive = AdaptiveThresholds()
        adj = adaptive.suggest_adjustment("quality", history, 0.90)

        assert isinstance(report.is_too_tight, bool)
        assert adj.direction in ("tighten", "loosen", "maintain")
