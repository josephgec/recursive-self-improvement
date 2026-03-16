"""Integration tests: full pipeline with all constraints, enforcement, monitoring, analysis."""

import pytest
from tests.conftest import MockAgent
from src.checker.suite import ConstraintSuite
from src.checker.runner import ConstraintRunner
from src.checker.cache import ConstraintCache
from src.enforcement.gate import ConstraintGate
from src.enforcement.rollback_trigger import RollbackTrigger
from src.enforcement.rejection_handler import RejectionHandler
from src.enforcement.audit import ConstraintAuditLog
from src.constraints.base import CheckContext, ConstraintResult
from src.constraints.custom import CustomConstraint
from src.monitoring.headroom import HeadroomMonitor
from src.monitoring.trend import TrendDetector
from src.monitoring.dashboard import DashboardConfig
from src.analysis.rejection_analysis import RejectionAnalyzer
from src.analysis.constraint_tightness import ConstraintTightnessAnalyzer
from src.analysis.report import generate_report
from src.evaluation.held_out_suite import HeldOutSuite
from src.evaluation.safety_suite import SafetySuite
from src.evaluation.diversity_probes import DiversityProbes
from src.evaluation.regression_suite import RegressionSuite


class _MockRollbackManager:
    def __init__(self):
        self.rollbacks = []

    def rollback(self, reason: str):
        self.rollbacks.append(reason)


class TestFullPipeline:
    """Full integration tests exercising all 7 constraints + enforcement."""

    def test_all_constraints_pass_allows(self, check_context):
        """Good agent: all 7 constraints pass, gate allows."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        gate = ConstraintGate(runner)

        agent = MockAgent()
        decision = gate.check_and_gate(agent, check_context)

        assert decision.allowed is True
        assert decision.verdict.passed is True
        assert len(decision.verdict.results) == 7

    def test_fail_accuracy_rejects_and_rollback(self, check_context):
        """Failing accuracy causes rejection and rollback."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        gate = ConstraintGate(runner)
        trigger = RollbackTrigger()
        manager = _MockRollbackManager()
        trigger.set_rollback_manager(manager)
        handler = RejectionHandler()
        audit = ConstraintAuditLog()

        agent = MockAgent(accuracy=0.50)
        decision = gate.check_and_gate(agent, check_context)

        assert decision.allowed is False
        assert "accuracy_floor" in decision.verdict.violations

        # Rejection handling
        msg = handler.handle(decision.verdict, check_context)
        assert "MODIFICATION REJECTED" in msg

        # Rollback
        trigger.trigger(decision.reason)
        assert len(manager.rollbacks) == 1

        # Audit
        audit.log(decision.verdict, check_context, "rejected")
        assert len(audit.get_violations()) == 1
        assert audit.verify_integrity()

    def test_fail_safety_rejects(self, check_context):
        """Failing safety causes rejection."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        gate = ConstraintGate(runner)

        agent = MockAgent(safety_pass_rate=0.5)
        decision = gate.check_and_gate(agent, check_context)

        assert decision.allowed is False
        assert "safety_eval" in decision.verdict.violations

    def test_headroom_monitoring(self, check_context):
        """Headroom monitor identifies at-risk constraints."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        agent = MockAgent()

        verdict = runner.run(agent, check_context)
        monitor = HeadroomMonitor(warning_threshold=0.05)
        report = monitor.compute_all(verdict)

        assert len(report.headrooms) == 7
        # Headrooms should all be present
        for name in verdict.results:
            assert name in report.headrooms

        # Dashboard renders
        dashboard = monitor.plot_headroom_dashboard(report)
        assert "HEADROOM DASHBOARD" in dashboard

    def test_trend_prediction(self, check_context):
        """Trend detector predicts violations from degrading headroom."""
        detector = TrendDetector(window_size=5)

        # Simulate degrading headroom
        for i in range(6):
            detector.record({"accuracy_floor": 0.10 - i * 0.02})

        trends = detector.compute_trends()
        assert "accuracy_floor" in trends
        acc_trend = trends["accuracy_floor"]
        assert acc_trend.direction == "degrading"
        assert acc_trend.slope < 0

        # Predict violation
        predicted = detector.predict_violation("accuracy_floor")
        assert predicted is not None
        assert predicted >= 0

        # Early warning
        warnings = detector.early_warning()
        assert len(warnings) > 0

    def test_rejection_analysis(self, check_context):
        """Rejection analyzer computes rates from audit entries."""
        audit = ConstraintAuditLog()
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)

        # Log some passed and failed runs
        for acc in [0.85, 0.85, 0.50, 0.85, 0.50]:
            agent = MockAgent(accuracy=acc)
            verdict = runner.run(agent, check_context)
            decision = "allowed" if verdict.passed else "rejected"
            audit.log(verdict, check_context, decision)

        analyzer = RejectionAnalyzer(audit.get_history())
        rate = analyzer.rejection_rate()
        assert 0.0 < rate < 1.0

        by_constraint = analyzer.rejection_by_constraint()
        assert "accuracy_floor" in by_constraint

        by_type = analyzer.rejection_by_modification_type()
        assert "test" in by_type

    def test_full_report_generation(self, check_context):
        """Full report generates all sections."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        agent = MockAgent()
        verdict = runner.run(agent, check_context)

        audit = ConstraintAuditLog()
        audit.log(verdict, check_context, "allowed")

        detector = TrendDetector(window_size=5)
        for name, result in verdict.results.items():
            detector.record({name: result.headroom})

        report = generate_report(
            verdict=verdict,
            audit_entries=audit.get_history(),
            trend_detector=detector,
        )

        assert "overall_passed" in report
        assert report["overall_passed"] is True
        assert "constraint_satisfaction" in report
        assert "headroom" in report
        assert "rejections" in report
        assert "tightness" in report
        assert "trends" in report

    def test_drift_ceiling_constraint(self, check_context):
        """Drift ceiling rejects high-drift agents."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)

        # High drift
        agent = MockAgent(drift=0.60)
        verdict = runner.run(agent, check_context)
        assert "drift_ceiling" in verdict.violations

        # Low drift
        agent2 = MockAgent(drift=0.10)
        verdict2 = runner.run(agent2, check_context)
        assert "drift_ceiling" not in verdict2.violations

    def test_regression_guard_constraint(self, check_context):
        """Regression guard rejects large regressions."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)

        # Large regression
        agent = MockAgent(regression=5.0)
        verdict = runner.run(agent, check_context)
        assert "regression_guard" in verdict.violations

        # Small regression
        agent2 = MockAgent(regression=1.0)
        verdict2 = runner.run(agent2, check_context)
        assert "regression_guard" not in verdict2.violations

    def test_consistency_constraint(self, check_context):
        """Consistency constraint rejects inconsistent agents."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)

        # Low consistency
        agent = MockAgent(consistency=0.50)
        verdict = runner.run(agent, check_context)
        assert "consistency" in verdict.violations

        # High consistency
        agent2 = MockAgent(consistency=0.95)
        verdict2 = runner.run(agent2, check_context)
        assert "consistency" not in verdict2.violations

    def test_latency_ceiling_constraint(self, check_context):
        """Latency ceiling rejects slow agents."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)

        # High latency
        agent = MockAgent(latency_p95=50000)
        verdict = runner.run(agent, check_context)
        assert "latency_ceiling" in verdict.violations

        # Low latency
        agent2 = MockAgent(latency_p95=5000)
        verdict2 = runner.run(agent2, check_context)
        assert "latency_ceiling" not in verdict2.violations

    def test_custom_constraint_integration(self, check_context):
        """Custom constraints can be added and evaluated."""
        suite = ConstraintSuite()
        custom = CustomConstraint(
            name="custom_check",
            description="Always passes",
            category="custom",
            threshold=0.0,
            check_fn=lambda a, c: ConstraintResult(True, 1.0, 0.0, 1.0),
        )
        suite = suite.add_custom(custom)
        runner = ConstraintRunner(suite, parallel=False)
        agent = MockAgent()

        verdict = runner.run(agent, check_context)
        assert "custom_check" in verdict.results
        assert verdict.results["custom_check"].satisfied is True

    def test_evaluation_suites_load(self):
        """All evaluation suites can be loaded."""
        held_out = HeldOutSuite().load(n=20)
        assert len(held_out) == 20

        safety = SafetySuite().load(n=10)
        assert len(safety) == 10

        probes = DiversityProbes().generate_probes()
        assert len(probes) > 0

        regression = RegressionSuite().load()
        assert len(regression) > 0

    def test_dashboard_config(self):
        """Dashboard config returns panel definitions."""
        config = DashboardConfig()
        panels = config.get_constraint_panels()
        assert len(panels) == 4
        assert any(p["title"] == "Quality Constraints" for p in panels)
        assert any(p["title"] == "Safety Constraints" for p in panels)

    def test_constraint_tightness_analysis(self, check_context):
        """Tightness analyzer detects too-tight and too-loose constraints."""
        # Build audit entries where accuracy_floor is violated > 50%
        entries = []
        for i in range(10):
            entries.append({
                "passed": i >= 6,
                "violations": ["accuracy_floor"] if i < 6 else [],
                "modification_type": "test",
                "results_summary": {
                    "accuracy_floor": {
                        "satisfied": i >= 6,
                        "measured_value": 0.78 if i < 6 else 0.85,
                        "threshold": 0.80,
                        "headroom": -0.02 if i < 6 else 0.05,
                    },
                    "safety_eval": {
                        "satisfied": True,
                        "measured_value": 1.0,
                        "threshold": 1.0,
                        "headroom": 0.0,
                    },
                },
            })

        analyzer = ConstraintTightnessAnalyzer(entries)
        analysis = analyzer.analyze()

        assert analysis["accuracy_floor"]["assessment"] == "too_tight"
        assert analysis["safety_eval"]["assessment"] == "too_loose"

        suggestions = analyzer.suggest_adjustments()
        constraint_names = [s["constraint"] for s in suggestions]
        assert "accuracy_floor" in constraint_names
        assert "safety_eval" in constraint_names

    def test_trend_stable_no_warning(self):
        """Stable trends produce no early warnings."""
        detector = TrendDetector(window_size=5)
        for _ in range(5):
            detector.record({"accuracy_floor": 0.10})

        warnings = detector.early_warning()
        assert len(warnings) == 0

    def test_trend_improving(self):
        """Improving trends are detected."""
        detector = TrendDetector(window_size=5)
        for i in range(5):
            detector.record({"accuracy_floor": 0.05 + i * 0.02})

        trends = detector.compute_trends()
        assert trends["accuracy_floor"].direction == "improving"

    def test_audit_integrity_across_mixed_decisions(self, check_context):
        """Hash chain remains valid across mixed allow/reject decisions."""
        audit = ConstraintAuditLog()
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)

        for acc in [0.85, 0.50, 0.85, 0.50, 0.85, 0.50, 0.85]:
            agent = MockAgent(accuracy=acc)
            verdict = runner.run(agent, check_context)
            decision = "allowed" if verdict.passed else "rejected"
            audit.log(verdict, check_context, decision)

        assert len(audit.get_history()) == 7
        assert audit.verify_integrity() is True

    def test_headroom_report_summary(self, check_context):
        """HeadroomReport summary is a readable string."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        agent = MockAgent()
        verdict = runner.run(agent, check_context)

        monitor = HeadroomMonitor(warning_threshold=0.05)
        report = monitor.compute_all(verdict)
        summary = report.summary()

        assert "Headroom Report" in summary

    def test_diversity_probes_compute_entropy(self):
        """DiversityProbes.compute_entropy works on sample outputs."""
        outputs = [
            "the quick brown fox jumps over the lazy dog",
            "a bright red bird sings in the tall green tree",
            "cold water flows down the deep mountain river",
        ]
        metrics = DiversityProbes.compute_entropy(outputs)
        assert metrics["token_entropy"] > 0
        assert metrics["vocab_size"] > 0

    def test_regression_suite_benchmarks(self):
        """RegressionSuite has all expected benchmarks."""
        suite = RegressionSuite()
        names = suite.get_benchmark_names()
        assert "mmlu" in names
        assert "hellaswag" in names
        assert len(names) == 6

        tasks = suite.load()
        for bench in names:
            assert bench in tasks
            assert len(tasks[bench]) == 10

    def test_constraint_repr(self):
        """Constraint repr is readable."""
        from src.constraints.accuracy_floor import AccuracyFloorConstraint
        c = AccuracyFloorConstraint(threshold=0.80)
        r = repr(c)
        assert "AccuracyFloorConstraint" in r
        assert "accuracy_floor" in r

    def test_multiple_failures(self, check_context):
        """Multiple constraint failures are all reported."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        gate = ConstraintGate(runner)

        # Bad everything
        agent = MockAgent(accuracy=0.50, safety_pass_rate=0.5, drift=0.80)
        decision = gate.check_and_gate(agent, check_context)

        assert decision.allowed is False
        violations = decision.verdict.violations
        assert len(violations) >= 3
