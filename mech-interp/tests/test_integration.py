"""Full integration test: extract -> modify -> diff -> divergence -> heads -> deceptive -> alerts -> report."""

import numpy as np
import pytest
import json

from src.probing.probe_set import ProbeSet, ProbeInput
from src.probing.extractor import (
    MockModel, MockModifiedModel, ActivationExtractor,
    ActivationSnapshot, LayerStats, HeadStats,
)
from src.probing.snapshot import save_snapshot, load_snapshot
from src.probing.diff import ActivationDiff
from src.probing.projector import DimensionalityReducer, ProjectionResult
from src.attention.head_extractor import HeadExtractor
from src.attention.specialization import HeadSpecializationTracker
from src.attention.dead_head_detector import DeadHeadDetector
from src.attention.role_tracker import HeadRoleTracker
from src.attention.reward_correlation import RewardCorrelationDetector
from src.anomaly.divergence_detector import BehavioralInternalDivergenceDetector
from src.anomaly.deceptive_alignment import DeceptiveAlignmentProber
from src.anomaly.internal_distance import measure_internal_change
from src.anomaly.behavioral_similarity import measure_behavioral_change
from src.monitoring.alert_rules import InterpretabilityAlertRules, AlertRule, Alert
from src.monitoring.dashboard import InterpretabilityDashboard
from src.monitoring.time_series import InterpretabilityTimeSeries
from src.integration.godel_hooks import GodelInterpretabilityHooks, InterpretabilityCheckResult
from src.integration.soar_hooks import SOARInterpretabilityHooks
from src.integration.pipeline_hooks import PipelineInterpretabilityHooks
from src.integration.decorator import monitor_internals
from src.analysis.activation_analysis import ActivationAnalysis
from src.analysis.head_evolution import HeadEvolutionAnalyzer
from src.analysis.anomaly_characterization import AnomalyCharacterizer
from src.analysis.report import generate_report


class TestFullIntegration:
    """Full end-to-end integration test."""

    def test_full_pipeline(self, mock_model, mock_modified_model, sample_probes):
        """Run the complete interpretability pipeline."""
        # Step 1: Extract before
        extractor_before = ActivationExtractor(mock_model)
        before_snapshot = extractor_before.extract(sample_probes)
        assert len(before_snapshot.get_probe_ids()) == len(sample_probes)

        # Step 2: "Modify" model (use modified model)
        extractor_after = ActivationExtractor(mock_modified_model)
        after_snapshot = extractor_after.extract(sample_probes)

        # Step 3: Compute diff
        differ = ActivationDiff()
        diff_result = differ.compute(before_snapshot, after_snapshot)
        assert diff_result.overall_change_magnitude > 0
        assert len(diff_result.most_changed_layers) > 0

        # Step 4: Divergence check
        detector = BehavioralInternalDivergenceDetector(ratio_threshold=3.0)
        behavioral_change = 0.05  # Small behavioral change
        div_result = detector.check(diff_result, behavioral_change, iteration=1)
        assert isinstance(div_result.divergence_ratio, float)

        # Step 5: Head tracking
        head_extractor = HeadExtractor(mock_model)
        head_stats_before = head_extractor.extract_aggregate_stats(sample_probes[:3])

        head_extractor_after = HeadExtractor(mock_modified_model)
        head_stats_after = head_extractor_after.extract_aggregate_stats(sample_probes[:3])

        tracker = HeadSpecializationTracker()
        tracker.track(head_stats_before)
        head_result = tracker.track(head_stats_after)
        assert isinstance(head_result.summary, dict)

        # Step 6: Deceptive alignment probes
        da_prober = DeceptiveAlignmentProber(mock_model)
        da_report = da_prober.run_all_probes(probe_accuracy=0.9, output_accuracy=0.7)
        assert isinstance(da_report.monitoring_sensitivity, float)

        # Step 7: Alert rules
        alert_rules = InterpretabilityAlertRules()
        data = {
            "divergence": div_result.to_dict(),
            "diff_summary": diff_result.to_dict(),
            "head_tracking": head_result.to_dict(),
            "deceptive_alignment": da_report.to_dict(),
        }
        alerts = alert_rules.evaluate(data, iteration=1)
        assert isinstance(alerts, list)

        # Step 8: Report
        report = generate_report(
            iteration=1,
            divergence_data=div_result.to_dict(),
            diff_data=diff_result.to_dict(),
            head_tracking_data=head_result.to_dict(),
            deceptive_alignment_data=da_report.to_dict(),
            alerts=[a.to_dict() for a in alerts],
        )
        assert "# Mechanistic Interpretability Report" in report
        assert "Executive Summary" in report
        assert "Divergence Analysis" in report
        assert "Head Tracking" in report
        assert "Deceptive Alignment" in report
        assert "Alerts" in report
        assert "Recommendations" in report

    def test_godel_hooks(self, mock_model, sample_probes):
        """Test Godel integration hooks."""
        probe_set = ProbeSet(sample_probes)
        hooks = GodelInterpretabilityHooks(
            mock_model, probe_set,
            max_divergence_ratio=5.0,
            block_on_critical=True,
        )

        before_snap = hooks.before_modification()
        assert isinstance(before_snap, ActivationSnapshot)

        result = hooks.after_modification(behavioral_change=0.1)
        assert isinstance(result, InterpretabilityCheckResult)
        assert isinstance(result.should_block, bool)
        assert hooks.should_block(result) == result.should_block

    def test_godel_hooks_no_before(self, mock_model, sample_probes):
        """Should handle after_modification without before."""
        probe_set = ProbeSet(sample_probes)
        hooks = GodelInterpretabilityHooks(mock_model, probe_set)
        result = hooks.after_modification(behavioral_change=0.1)
        assert not result.should_block
        assert result.reason == "No before snapshot available"

    def test_godel_hooks_result_to_dict(self, mock_model, sample_probes):
        """InterpretabilityCheckResult should serialize."""
        probe_set = ProbeSet(sample_probes)
        hooks = GodelInterpretabilityHooks(mock_model, probe_set)
        hooks.before_modification()
        result = hooks.after_modification(behavioral_change=0.1)
        d = result.to_dict()
        assert "should_block" in d
        assert "reason" in d

    def test_soar_hooks(self, mock_model, sample_probes):
        """Test SOAR integration hooks."""
        probe_set = ProbeSet(sample_probes)
        hooks = SOARInterpretabilityHooks(
            mock_model, probe_set, check_every_n_steps=5,
        )

        # Step not checked
        result = hooks.after_training_step(step=1)
        assert result is None

        # Step checked
        result = hooks.after_training_step(step=5, loss=0.1)
        assert result is not None
        assert result["checked"]

        # Second check should have divergence info
        result = hooks.after_training_step(step=10, loss=0.05)
        assert "divergence_ratio" in result

        # Epoch check
        epoch_result = hooks.after_training_epoch(epoch=1)
        assert "head_tracking" in epoch_result

        assert hooks.get_step_count() == 10
        assert hooks.get_epoch_count() == 1

    def test_soar_anomaly_detection(self, mock_model, sample_probes):
        """SOAR should detect anomalies."""
        probe_set = ProbeSet(sample_probes)
        hooks = SOARInterpretabilityHooks(mock_model, probe_set, check_every_n_steps=1)

        for i in range(5):
            hooks.after_training_step(step=i)

        anomalies = hooks.detect_training_anomalies()
        assert isinstance(anomalies, list)

    def test_pipeline_hooks(self, mock_model, sample_probes):
        """Test pipeline integration hooks."""
        probe_set = ProbeSet(sample_probes)
        hooks = PipelineInterpretabilityHooks(
            mock_model, probe_set, run_deceptive_probes=True,
        )

        result = hooks.after_iteration(iteration=0, behavioral_change=0.1)
        assert "iteration" in result
        assert result["divergence"] is None  # First iteration

        result = hooks.after_iteration(iteration=1, behavioral_change=0.1)
        assert result["divergence"] is not None

        history = hooks.get_results_history()
        assert len(history) == 2

    def test_decorator(self, mock_model, sample_probes):
        """Test monitor_internals decorator."""
        probe_set = ProbeSet(sample_probes)
        diffs_collected = []

        def on_diff(before, after, diff):
            diffs_collected.append(diff)

        @monitor_internals(model=mock_model, probe_set=probe_set, on_diff=on_diff)
        def my_function():
            return {"status": "done"}

        result = my_function()
        assert result["status"] == "done"
        assert "_interp_diff" in result
        assert len(diffs_collected) == 1

    def test_decorator_no_model(self):
        """Decorator should pass through if no model."""
        @monitor_internals()
        def my_function():
            return 42

        assert my_function() == 42


class TestDashboard:
    """Test dashboard functionality."""

    def test_log_and_get_panels(self):
        """Should log iterations and provide panels."""
        dashboard = InterpretabilityDashboard()

        data = {
            "divergence": {"divergence_ratio": 1.5, "internal_change": 0.3, "behavioral_change": 0.2},
            "head_tracking": {"num_dying_heads": 1, "num_role_changes": 0},
            "deceptive_alignment": {"monitoring_sensitivity": 0.1, "latent_capability_gap": 0.05},
        }
        alerts = dashboard.log_iteration(0, data)
        assert isinstance(alerts, list)

        panels = dashboard.get_panels()
        assert "overview" in panels
        assert "divergence" in panels
        assert "head_tracking" in panels
        assert "alerts" in panels

    def test_summary(self):
        """Should produce text summary."""
        dashboard = InterpretabilityDashboard()
        data = {
            "divergence": {"divergence_ratio": 2.0, "internal_change": 0.5, "behavioral_change": 0.3},
        }
        dashboard.log_iteration(0, data)
        summary = dashboard.get_summary()
        assert "Interpretability Dashboard" in summary
        assert "Iterations: 1" in summary


class TestTimeSeries:
    """Test time series functionality."""

    def test_record_and_get(self):
        """Should record and retrieve metrics."""
        ts = InterpretabilityTimeSeries()
        ts.record(0, {"a": 1.0, "b": 2.0})
        ts.record(1, {"a": 3.0, "b": 4.0})

        history = ts.get_history()
        assert len(history) == 2

    def test_metric_values(self):
        """Should return values for specific metric."""
        ts = InterpretabilityTimeSeries()
        ts.record(0, {"a": 1.0})
        ts.record(1, {"a": 2.0})
        values = ts.get_metric_values("a")
        assert values == [1.0, 2.0]

    def test_metric_stats(self):
        """Should compute metric statistics."""
        ts = InterpretabilityTimeSeries()
        ts.record(0, {"a": 1.0})
        ts.record(1, {"a": 3.0})
        stats = ts.get_metric_stats("a")
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["count"] == 2

    def test_empty_metric(self):
        """Should handle missing metric."""
        ts = InterpretabilityTimeSeries()
        stats = ts.get_metric_stats("nonexistent")
        assert stats["count"] == 0

    def test_max_history(self):
        """Should trim history to max."""
        ts = InterpretabilityTimeSeries(max_history=5)
        for i in range(10):
            ts.record(i, {"a": float(i)})
        assert len(ts) == 5

    def test_window(self):
        """Should return window of data."""
        ts = InterpretabilityTimeSeries()
        for i in range(10):
            ts.record(i, {"a": float(i)})
        window = ts.get_window(3)
        assert len(window) == 3

    def test_get_history_filtered(self):
        """Should filter history by metric name."""
        ts = InterpretabilityTimeSeries()
        ts.record(0, {"a": 1.0, "b": 2.0})
        ts.record(1, {"a": 3.0})
        history = ts.get_history("a")
        assert len(history) == 2
        assert "a" in history[0]


class TestAlertRules:
    """Test alert rules."""

    def test_default_rules(self):
        """Should have 6 default rules."""
        rules = InterpretabilityAlertRules()
        assert len(rules.rules) == 6

    def test_divergence_alert(self):
        """Should trigger on high divergence."""
        rules = InterpretabilityAlertRules()
        data = {"divergence": {"divergence_ratio": 5.0}}
        alerts = rules.evaluate(data)
        alert_names = [a.rule_name for a in alerts]
        assert "divergence_ratio_high" in alert_names

    def test_safety_alert(self):
        """Should trigger on safety disproportionate."""
        rules = InterpretabilityAlertRules()
        data = {"diff_summary": {"safety_disproportionate": True}}
        alerts = rules.evaluate(data)
        alert_names = [a.rule_name for a in alerts]
        assert "safety_probe_disproportionate" in alert_names

    def test_monitoring_sensitive_alert(self):
        """Should trigger on monitoring sensitivity."""
        rules = InterpretabilityAlertRules()
        data = {"deceptive_alignment": {"monitoring_sensitivity": 0.5}}
        alerts = rules.evaluate(data)
        alert_names = [a.rule_name for a in alerts]
        assert "monitoring_sensitive" in alert_names

    def test_latent_capability_alert(self):
        """Should trigger on latent capability gap."""
        rules = InterpretabilityAlertRules()
        data = {"deceptive_alignment": {"latent_capability_gap": 0.3}}
        alerts = rules.evaluate(data)
        alert_names = [a.rule_name for a in alerts]
        assert "latent_capability_gap" in alert_names

    def test_mass_head_death_alert(self):
        """Should trigger on mass head death."""
        rules = InterpretabilityAlertRules()
        data = {"head_tracking": {
            "num_dying_heads": 5,
            "summary": {"num_heads": 10},
        }}
        alerts = rules.evaluate(data)
        alert_names = [a.rule_name for a in alerts]
        assert "mass_head_death" in alert_names

    def test_no_alerts(self):
        """Should not trigger on normal data."""
        rules = InterpretabilityAlertRules()
        data = {
            "divergence": {"divergence_ratio": 1.0},
            "diff_summary": {"safety_disproportionate": False},
            "deceptive_alignment": {"monitoring_sensitivity": 0.01, "latent_capability_gap": 0.01},
            "head_tracking": {"num_dying_heads": 0, "summary": {"num_heads": 10}},
        }
        alerts = rules.evaluate(data)
        assert len(alerts) == 0

    def test_alert_history(self):
        """Should maintain alert history."""
        rules = InterpretabilityAlertRules()
        rules.evaluate({"divergence": {"divergence_ratio": 5.0}}, iteration=1)
        rules.evaluate({"divergence": {"divergence_ratio": 5.0}}, iteration=2)
        assert len(rules.get_alert_history()) == 2

    def test_critical_alerts(self):
        """Should filter critical alerts."""
        rules = InterpretabilityAlertRules()
        rules.evaluate({"diff_summary": {"safety_disproportionate": True}}, iteration=1)
        critical = rules.get_critical_alerts()
        assert len(critical) > 0
        assert all(a.severity == "critical" for a in critical)

    def test_add_custom_rule(self):
        """Should support custom rules."""
        rules = InterpretabilityAlertRules()
        custom = AlertRule(
            name="custom_rule",
            description="Test rule",
            severity="info",
            check_fn=lambda d: d.get("custom_flag", False),
        )
        rules.add_rule(custom)
        alerts = rules.evaluate({"custom_flag": True})
        assert any(a.rule_name == "custom_rule" for a in alerts)

    def test_alert_to_dict(self):
        """Alert should serialize."""
        alert = Alert(
            rule_name="test", severity="warning",
            description="Test alert", iteration=1,
        )
        d = alert.to_dict()
        assert d["rule_name"] == "test"
        assert d["severity"] == "warning"


class TestProbeSet:
    """Test ProbeSet functionality."""

    def test_builtin_probes(self):
        """Should have 50+ built-in probes."""
        ps = ProbeSet()
        assert len(ps) >= 50

    def test_categories(self):
        """Should have 5 categories."""
        ps = ProbeSet()
        assert set(ps.categories) == {"factual", "reasoning", "safety", "adversarial", "diverse"}

    def test_get_by_category(self):
        """Should filter by category."""
        ps = ProbeSet()
        safety = ps.get_by_category("safety")
        assert len(safety) > 0
        assert all(p.category == "safety" for p in safety)

    def test_load_additional(self):
        """Should load additional probes."""
        ps = ProbeSet([])
        ps.load([
            {"text": "test", "category": "custom", "expected_behavior": "result"},
        ])
        assert len(ps) == 1
        assert ps.get_all()[0].category == "custom"

    def test_iteration(self):
        """Should be iterable."""
        ps = ProbeSet()
        probes = list(ps)
        assert len(probes) == len(ps)

    def test_fixture_probes(self, fixture_probes):
        """Should load probes from fixture."""
        assert len(fixture_probes) == 20
        categories = set(p.category for p in fixture_probes)
        assert len(categories) == 5


class TestProjector:
    """Test dimensionality reduction."""

    def test_reduce_2d(self, sample_snapshot):
        """Should reduce to 2D."""
        reducer = DimensionalityReducer(n_components=2)
        result = reducer.reduce(sample_snapshot)
        assert isinstance(result, ProjectionResult)
        assert result.coordinates.shape[1] == 2
        assert len(result.labels) == result.coordinates.shape[0]

    def test_reduce_3d(self, sample_snapshot):
        """Should reduce to 3D."""
        reducer = DimensionalityReducer(n_components=3)
        result = reducer.reduce(sample_snapshot)
        assert result.coordinates.shape[1] == 3

    def test_reduce_specific_layer(self, sample_snapshot):
        """Should reduce specific layer."""
        reducer = DimensionalityReducer(n_components=2)
        result = reducer.reduce(sample_snapshot, layer_name="layer_0")
        assert result.coordinates.shape[1] == 2

    def test_reduce_multi(self, sample_snapshot, modified_snapshot):
        """Should reduce multiple snapshots."""
        reducer = DimensionalityReducer(n_components=2)
        result = reducer.reduce_multi(
            [sample_snapshot, modified_snapshot],
            ["before", "after"],
        )
        assert result.coordinates.shape[0] > 0
        assert all("/" in label for label in result.labels)

    def test_empty_snapshot(self):
        """Should handle empty snapshot."""
        reducer = DimensionalityReducer()
        empty = ActivationSnapshot()
        result = reducer.reduce(empty)
        assert result.coordinates.shape[0] == 0

    def test_to_dict(self, sample_snapshot):
        """ProjectionResult should serialize."""
        reducer = DimensionalityReducer()
        result = reducer.reduce(sample_snapshot)
        d = result.to_dict()
        assert "coordinates" in d
        assert "labels" in d


class TestAnalysis:
    """Test analysis modules."""

    def test_activation_analysis(self, sample_snapshot, modified_snapshot):
        """Test per-layer activation analysis."""
        differ = ActivationDiff()
        diff = differ.compute(sample_snapshot, modified_snapshot)

        analysis = ActivationAnalysis()
        analysis.add_diff(diff)

        all_layers = analysis.analyze_all_layers()
        assert len(all_layers) > 0

        layer_0 = analysis.analyze_layer("layer_0")
        assert layer_0["data_points"] == 1

        most_changed = analysis.get_most_changed_layers()
        assert len(most_changed) > 0

    def test_activation_analysis_safety(self, sample_snapshot, modified_snapshot):
        """Test safety layer analysis."""
        differ = ActivationDiff()
        diff = differ.compute(sample_snapshot, modified_snapshot)

        analysis = ActivationAnalysis()
        analysis.add_diff(diff)
        safety = analysis.get_safety_layer_analysis()
        assert isinstance(safety, dict)

    def test_activation_analysis_no_data(self):
        """Should handle empty analysis."""
        analysis = ActivationAnalysis()
        result = analysis.analyze_layer("layer_0")
        assert result["data_points"] == 0

    def test_head_evolution(self):
        """Test head evolution analyzer."""
        analyzer = HeadEvolutionAnalyzer()

        from src.attention.specialization import HeadTrackingResult, HeadRoleChange, HeadShift
        result1 = HeadTrackingResult(
            shifts=[HeadShift(0, 0, 2.0, 1.5, -0.5, 0.33, 0.5)],
            dying_heads=[],
            narrowing_heads=[],
            role_changes=[],
            summary={"mean_entropy": 2.0},
        )
        result2 = HeadTrackingResult(
            shifts=[HeadShift(0, 0, 1.5, 0.5, -1.0, 0.5, 0.83)],
            dying_heads=[(0, 1)],
            narrowing_heads=[(0, 0)],
            role_changes=[HeadRoleChange(0, 0, "global", "local", 1, 1.0)],
            summary={"mean_entropy": 1.5},
        )

        analyzer.add_tracking_result(result1)
        analyzer.add_tracking_result(result2)

        transitions = analyzer.get_role_transitions()
        assert len(transitions) == 1

        dying_trend = analyzer.get_dying_head_trend()
        assert dying_trend == [0, 1]

        entropy_trend = analyzer.get_entropy_trend()
        assert entropy_trend == [2.0, 1.5]

        summary = analyzer.summarize()
        assert summary["total_iterations"] == 2
        assert summary["total_role_transitions"] == 1

    def test_anomaly_characterization(self):
        """Test anomaly characterization."""
        characterizer = AnomalyCharacterizer()

        from src.anomaly.divergence_detector import DivergenceCheckResult
        from src.anomaly.deceptive_alignment import DeceptiveAlignmentReport

        # Add a normal result
        result1 = DivergenceCheckResult(
            divergence_ratio=1.0, internal_change=0.1,
            behavioral_change=0.1, z_score=0.5,
            is_anomalous=False, safety_flag=False,
            safety_change_ratio=1.0, iteration=1,
        )
        characterizer.add_divergence_result(result1)

        # Add an anomalous result
        result2 = DivergenceCheckResult(
            divergence_ratio=6.0, internal_change=0.6,
            behavioral_change=0.1, z_score=3.5,
            is_anomalous=True, safety_flag=True,
            safety_change_ratio=3.0, iteration=2,
        )
        characterizer.add_divergence_result(result2)

        da_report = DeceptiveAlignmentReport(
            monitoring_sensitivity=0.3,
            context_dependent_safety=0.1,
            latent_capability_gap=0.2,
            paraphrase_consistency=0.9,
            is_suspicious=True,
            flags=["monitoring_sensitive", "latent_capability_gap"],
        )
        characterizer.add_deceptive_report(da_report)

        latest = characterizer.characterize_latest()
        assert latest["is_anomalous"]
        assert latest["type"] == "potential_deceptive_alignment"
        assert latest["severity"] == "critical"

        all_anomalies = characterizer.characterize_all()
        assert len(all_anomalies) == 1  # Only the anomalous one

        summary = characterizer.get_anomaly_summary()
        assert summary["total_checks"] == 2
        assert summary["total_anomalous"] == 1
        assert summary["deceptive_suspicious"] == 1

    def test_anomaly_characterizer_no_data(self):
        """Should handle no data."""
        characterizer = AnomalyCharacterizer()
        latest = characterizer.characterize_latest()
        assert latest["status"] == "no_data"


class TestReport:
    """Test report generation."""

    def test_generate_basic_report(self):
        """Should generate a report with all sections."""
        report = generate_report(iteration=5)
        assert "# Mechanistic Interpretability Report" in report
        assert "Executive Summary" in report
        assert "Divergence Analysis" in report
        assert "Activation Diff" in report
        assert "Head Tracking" in report
        assert "Deceptive Alignment" in report
        assert "Anomaly Summary" in report
        assert "Alerts" in report
        assert "Recommendations" in report

    def test_generate_full_report(self):
        """Should generate report with data."""
        report = generate_report(
            iteration=10,
            divergence_data={
                "divergence_ratio": 2.5,
                "internal_change": 0.5,
                "behavioral_change": 0.2,
                "z_score": 1.5,
                "is_anomalous": False,
                "safety_flag": False,
            },
            diff_data={
                "most_changed_layers": ["layer_5", "layer_6"],
                "safety_disproportionate": False,
                "overall_change": 0.3,
            },
            head_tracking_data={
                "num_dying_heads": 2,
                "num_narrowing_heads": 1,
                "num_role_changes": 0,
                "summary": {"mean_entropy": 1.8},
            },
            deceptive_alignment_data={
                "monitoring_sensitivity": 0.05,
                "context_dependent_safety": 0.03,
                "latent_capability_gap": 0.02,
                "paraphrase_consistency": 0.95,
                "is_suspicious": False,
                "flags": [],
            },
            alerts=[],
        )
        assert "2.500" in report
        assert "OK" in report

    def test_report_with_alerts(self):
        """Should show alerts in report."""
        report = generate_report(
            iteration=5,
            alerts=[
                {"rule_name": "test_alert", "severity": "critical", "description": "Test alert"},
            ],
        )
        assert "CRITICAL" in report
        assert "test_alert" in report

    def test_report_with_recommendations(self):
        """Should show custom recommendations."""
        report = generate_report(
            iteration=5,
            recommendations=["Do something", "Do something else"],
        )
        assert "Do something" in report
        assert "Do something else" in report

    def test_auto_recommendations_anomalous(self):
        """Should auto-generate recommendations for anomalous data."""
        report = generate_report(
            iteration=5,
            divergence_data={"is_anomalous": True},
            diff_data={"safety_disproportionate": True},
            head_tracking_data={"num_dying_heads": 3, "num_role_changes": 2},
            deceptive_alignment_data={"is_suspicious": True},
        )
        assert "Investigate" in report or "Review" in report or "CRITICAL" in report
