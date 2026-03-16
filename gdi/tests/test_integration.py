"""Full integration tests for GDI system."""

import json
import os
import tempfile

import pytest

from src.composite.gdi import GoalDriftIndex
from src.composite.weights import WeightConfig
from src.reference.collector import ReferenceCollector, ReferenceOutputs
from src.reference.store import ReferenceStore
from src.reference.updater import ReferenceUpdater, UpdateProposal
from src.calibration.collapse_calibrator import CollapseCalibrator
from src.alerting.alert_manager import AlertManager
from src.alerting.escalation import EscalationPolicy
from src.alerting.channels import LogChannel
from src.monitoring.time_series import GDITimeSeries
from src.monitoring.anomaly_detector import AnomalyDetector, Anomaly
from src.monitoring.dashboard_config import DashboardConfig
from src.analysis.signal_decomposition import SignalDecompositionAnalyzer
from src.analysis.drift_characterization import DriftCharacterizer
from src.analysis.early_warning import EarlyWarningAnalyzer
from src.analysis.report import generate_report

from tests.conftest import MockAgent


class TestFullPipeline:
    """Full end-to-end integration tests."""

    def test_collect_compute_alert(
        self, reference_outputs, drifted_outputs, collapsed_outputs, tmp_path
    ):
        """Full pipeline: collect reference, compute GDI, generate alerts."""
        # 1. Setup reference store
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        # 2. Create GDI instance
        gdi = GoalDriftIndex()

        # 3. No drift → green
        result_green = gdi.compute(reference_outputs, reference_outputs)
        assert result_green.alert_level == "green"
        assert result_green.composite_score < 0.15

        # 4. Minor drift → yellow or low
        result_drift = gdi.compute(drifted_outputs, reference_outputs)
        assert result_drift.composite_score > result_green.composite_score

        # 5. Major drift → should be elevated
        result_collapsed = gdi.compute(collapsed_outputs, reference_outputs)
        assert result_collapsed.composite_score > result_drift.composite_score
        assert result_collapsed.composite_score > 0.3

        # 6. Alert management
        channel = LogChannel()
        alert_mgr = AlertManager(channels=[channel])

        alert_mgr.process(result_green, 1)
        assert len(alert_mgr.alert_history) == 0  # Green = no alert

        alert = alert_mgr.process(result_collapsed, 2)
        assert alert is not None
        assert len(channel.sent_alerts) >= 1

    def test_calibrate_and_apply(
        self, reference_outputs, collapsed_outputs
    ):
        """Calibrate thresholds and apply to GDI."""
        gdi = GoalDriftIndex()
        calibrator = CollapseCalibrator()

        collapse_data = [
            {"outputs": reference_outputs, "reference": reference_outputs, "health": "healthy"},
            {"outputs": collapsed_outputs, "reference": reference_outputs, "health": "collapsed"},
        ]

        thresholds = calibrator.calibrate(gdi, collapse_data)

        # Apply calibrated thresholds
        gdi_calibrated = GoalDriftIndex(
            green_max=thresholds.green_max,
            yellow_max=thresholds.yellow_max,
            orange_max=thresholds.orange_max,
        )

        # Verify thresholds are applied
        assert gdi_calibrated.green_max == thresholds.green_max

    def test_reference_collection(self, reference_outputs):
        """Test reference collection pipeline."""
        collector = ReferenceCollector(samples_per_task=3)
        agent = MockAgent(reference_outputs)
        tasks = ["task1", "task2", "task3"]

        outputs = collector.collect(agent, tasks)

        assert isinstance(outputs, ReferenceOutputs)
        assert len(outputs.outputs) == 9  # 3 tasks × 3 samples
        assert len(outputs.task_outputs) == 3
        assert outputs.metadata["total_outputs"] == 9

    def test_reference_store_lifecycle(self, reference_outputs, tmp_path):
        """Test reference store save/load/update."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)

        # Initially doesn't exist
        assert not store.exists()

        # Save
        store.save({"outputs": reference_outputs, "version": 1})
        assert store.exists()

        # Load
        data = store.load()
        assert data["version"] == 1
        assert len(data["outputs"]) == len(reference_outputs)

        # Update (archives old)
        store.update({"outputs": reference_outputs, "version": 2})
        data = store.load()
        assert data["version"] == 2

    def test_reference_store_load_missing(self, tmp_path):
        """Loading missing store should raise."""
        store = ReferenceStore(str(tmp_path / "missing.json"))
        with pytest.raises(FileNotFoundError):
            store.load()

    def test_reference_updater(self, reference_outputs, tmp_path):
        """Test human-gated reference updater."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        collector = ReferenceCollector(samples_per_task=2)
        updater = ReferenceUpdater(store, collector)

        agent = MockAgent(reference_outputs)
        proposal = updater.propose_update(agent, ["task1", "task2"])

        assert isinstance(proposal, UpdateProposal)
        assert not proposal.approved
        assert len(updater.pending_proposals) == 1

        # Approve
        updater.approve_update(proposal, approved_by="test_user")
        assert proposal.approved
        assert proposal.approved_by == "test_user"
        assert len(updater.pending_proposals) == 0

    def test_time_series(self, reference_outputs, tmp_path):
        """Test time series recording and retrieval."""
        ts_path = str(tmp_path / "history.json")
        ts = GDITimeSeries(ts_path)

        gdi = GoalDriftIndex()
        result = gdi.compute(reference_outputs, reference_outputs)

        ts.record(result, iteration=0)
        ts.record(result, iteration=1)

        history = ts.get_history()
        assert len(history) == 2

        window = ts.get_window(1)
        assert len(window) == 1

        scores = ts.get_scores()
        assert len(scores) == 2

        exported = ts.export()
        assert len(exported) == 2

        # Verify persistence
        ts2 = GDITimeSeries(ts_path)
        assert len(ts2.get_history()) == 2

    def test_anomaly_detection(self):
        """Test anomaly detection on time series."""
        detector = AnomalyDetector(z_threshold=2.0, min_history=5)

        # Normal history with one spike
        history = [0.1, 0.12, 0.11, 0.1, 0.13, 0.9, 0.11, 0.1]
        anomalies = detector.detect(history)

        assert len(anomalies) > 0
        assert any(a.direction == "high" for a in anomalies)

    def test_anomaly_detection_insufficient(self):
        """Insufficient data should return no anomalies."""
        detector = AnomalyDetector(min_history=5)
        anomalies = detector.detect([0.1, 0.2])
        assert len(anomalies) == 0

    def test_anomaly_detection_flat(self):
        """Flat data should have no anomalies."""
        detector = AnomalyDetector()
        anomalies = detector.detect([0.5, 0.5, 0.5, 0.5, 0.5])
        assert len(anomalies) == 0

    def test_dashboard_config(self):
        """Dashboard config should return panels."""
        panels = DashboardConfig.get_panels()
        assert len(panels) > 0
        assert any(p["title"] == "GDI Composite Score" for p in panels)

    def test_signal_decomposition(self, reference_outputs, collapsed_outputs):
        """Test signal decomposition analysis."""
        gdi = GoalDriftIndex()
        result = gdi.compute(collapsed_outputs, reference_outputs)

        analyzer = SignalDecompositionAnalyzer()
        decomp = analyzer.decompose(result)

        assert "contributions" in decomp
        assert "primary_driver" in decomp
        assert decomp["primary_driver"] in (
            "semantic", "lexical", "structural", "distributional"
        )

    def test_signal_decomposition_history(self, reference_outputs, collapsed_outputs):
        """Test primary driver identification across history."""
        gdi = GoalDriftIndex()
        history = [
            gdi.compute(reference_outputs, reference_outputs),
            gdi.compute(collapsed_outputs, reference_outputs),
        ]

        analyzer = SignalDecompositionAnalyzer()
        driver = analyzer.identify_primary_driver(history)
        assert driver in ("semantic", "lexical", "structural", "distributional")

        trajectories = analyzer.plot_signal_trajectories(history)
        assert len(trajectories["composite"]) == 2

    def test_signal_decomposition_zero_score(self, reference_outputs):
        """Decomposition of zero score should handle gracefully."""
        gdi = GoalDriftIndex()
        result = gdi.compute(reference_outputs, reference_outputs)

        analyzer = SignalDecompositionAnalyzer()
        decomp = analyzer.decompose(result)
        assert "primary_driver" in decomp

    def test_signal_decomposition_empty_history(self):
        """Empty history should return 'none'."""
        analyzer = SignalDecompositionAnalyzer()
        assert analyzer.identify_primary_driver([]) == "none"

    def test_drift_characterization(self, reference_outputs, collapsed_outputs):
        """Test drift characterization."""
        gdi = GoalDriftIndex()
        result = gdi.compute(collapsed_outputs, reference_outputs)

        characterizer = DriftCharacterizer()
        char = characterizer.characterize(result)

        assert char.drift_type in (
            "reasoning_shift", "style_shift", "structural_shift",
            "distributional_shift", "collapse", "comprehensive", "minimal"
        )
        assert 0 <= char.confidence <= 1.0

    def test_drift_characterization_minimal(self, reference_outputs):
        """Minimal drift should be characterized as minimal."""
        gdi = GoalDriftIndex()
        result = gdi.compute(reference_outputs, reference_outputs)

        characterizer = DriftCharacterizer()
        char = characterizer.characterize(result)
        assert char.drift_type == "minimal"

    def test_early_warning(self):
        """Test early warning analysis."""
        analyzer = EarlyWarningAnalyzer()

        gdi_history = [0.1, 0.2, 0.5, 0.7, 0.9]
        accuracy_history = [0.95, 0.90, 0.85, 0.70, 0.50]

        result = analyzer.compute_lead_time(gdi_history, accuracy_history)
        assert result["lead_time_steps"] is not None

    def test_early_warning_empty(self):
        """Empty data should return appropriate result."""
        analyzer = EarlyWarningAnalyzer()
        result = analyzer.compute_lead_time([], [])
        assert result["lead_time_steps"] is None

    def test_early_warning_no_alert(self):
        """No GDI alert should report no lead time."""
        analyzer = EarlyWarningAnalyzer()
        result = analyzer.compute_lead_time(
            [0.1, 0.1, 0.1],
            [0.95, 0.90, 0.85],
        )
        assert result["lead_time_steps"] is None

    def test_early_warning_plot_data(self):
        """Should return plot data."""
        analyzer = EarlyWarningAnalyzer()
        data = analyzer.plot_early_warning([0.1, 0.5], [0.9, 0.7])
        assert "gdi_scores" in data
        assert "accuracy_scores" in data

    def test_generate_report(
        self, reference_outputs, drifted_outputs, collapsed_outputs
    ):
        """Test report generation."""
        gdi = GoalDriftIndex()
        history = [
            gdi.compute(reference_outputs, reference_outputs),
            gdi.compute(drifted_outputs, reference_outputs),
            gdi.compute(collapsed_outputs, reference_outputs),
        ]

        report = generate_report(
            gdi_history=history,
            accuracy_history=[0.95, 0.85, 0.40],
            alert_log=[{"level": "yellow", "iteration": 2}],
            calibration_info={"auc": 0.95},
        )

        assert "generated_at" in report
        assert "latest" in report
        assert "decomposition" in report
        assert "characterization" in report
        assert "early_warning" in report
        assert "calibration" in report
        assert "alert_log" in report
        assert "trajectory" in report
        assert "primary_driver" in report

    def test_generate_report_minimal(self):
        """Report with no data should still work."""
        report = generate_report()
        assert "generated_at" in report
        assert "title" in report

    def test_escalation_in_pipeline(
        self, reference_outputs, collapsed_outputs
    ):
        """Test escalation policy in a pipeline scenario."""
        gdi = GoalDriftIndex()
        escalation = EscalationPolicy(consecutive_red_limit=3)

        # Simulate pipeline iterations
        result_green = gdi.compute(reference_outputs, reference_outputs)
        action = escalation.get_action(result_green.alert_level)
        assert action == "none"

        result_red = gdi.compute(collapsed_outputs, reference_outputs)
        if result_red.alert_level == "red":
            action = escalation.get_action("red")
            assert action in ("pause", "emergency_stop")

    def test_time_series_no_persistence(self):
        """Time series without persistence should work in memory."""
        ts = GDITimeSeries()

        from src.composite.gdi import GDIResult
        result = GDIResult(
            composite_score=0.5,
            alert_level="orange",
            trend="stable",
            semantic_score=0.4,
            lexical_score=0.5,
            structural_score=0.3,
            distributional_score=0.6,
        )
        ts.record(result)
        assert len(ts.get_history()) == 1

    def test_full_monitoring_cycle(
        self, reference_outputs, drifted_outputs, collapsed_outputs, tmp_path
    ):
        """Full monitoring cycle: collect, compute, monitor, alert, report."""
        # Setup
        store_path = str(tmp_path / "ref.json")
        ts_path = str(tmp_path / "history.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        ts = GDITimeSeries(ts_path)
        channel = LogChannel()
        alert_mgr = AlertManager(channels=[channel])
        detector = AnomalyDetector(z_threshold=2.0, min_history=3)

        # Iteration 1: healthy
        r1 = gdi.compute(reference_outputs, reference_outputs)
        ts.record(r1, iteration=0)
        alert_mgr.process(r1, 0)

        # Iteration 2: drifted
        r2 = gdi.compute(drifted_outputs, reference_outputs)
        ts.record(r2, iteration=1)
        alert_mgr.process(r2, 1)

        # Iteration 3: collapsed
        r3 = gdi.compute(collapsed_outputs, reference_outputs)
        ts.record(r3, iteration=2)
        alert_mgr.process(r3, 2)

        # Verify trajectory
        assert len(ts.get_history()) == 3
        scores = ts.get_scores()
        assert scores[0] < scores[2]  # Drift increased

        # Anomaly detection
        anomalies = detector.detect(scores)
        # May or may not detect anomaly with only 3 points

        # Generate report
        report = generate_report(
            gdi_history=[r1, r2, r3],
            accuracy_history=[0.95, 0.85, 0.40],
            alert_log=[{"level": a.level, "iteration": a.iteration}
                       for a in alert_mgr.alert_history],
        )

        assert "latest" in report
        assert report["latest"]["composite_score"] == r3.composite_score
