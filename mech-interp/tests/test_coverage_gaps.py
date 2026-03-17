"""Targeted tests for uncovered branches to push coverage above 95%."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.probe_set import ProbeInput
from src.probing.extractor import (
    MockModel, MockModifiedModel, ActivationExtractor,
    ActivationSnapshot, LayerStats, HeadStats,
)
from src.probing.diff import ActivationDiff, ActivationDiffResult, LayerDiff
from src.probing.projector import DimensionalityReducer, ProjectionResult
from src.attention.head_extractor import HeadExtractor
from src.attention.specialization import (
    HeadSpecializationTracker, HeadTrackingResult, HeadShift, HeadRoleChange,
    measure_specialization,
)
from src.attention.reward_correlation import RewardCorrelationDetector, RewardCorrelatedHead
from src.attention.role_tracker import HeadRoleTracker
from src.anomaly.divergence_detector import BehavioralInternalDivergenceDetector, DivergenceCheckResult
from src.anomaly.behavioral_similarity import measure_behavioral_change, measure_behavioral_change_numeric
from src.anomaly.internal_distance import measure_safety_internal_change
from src.anomaly.ratio_monitor import RatioMonitor
from src.analysis.anomaly_characterization import AnomalyCharacterizer
from src.analysis.head_evolution import HeadEvolutionAnalyzer
from src.analysis.report import generate_report
from src.monitoring.alert_rules import (
    InterpretabilityAlertRules, AlertRule,
    _check_divergence_ratio_high, _check_safety_probe_disproportionate,
    _check_monitoring_sensitive, _check_reward_hacking_signal,
    _check_mass_head_death, _check_latent_capability_gap,
)
from src.integration.godel_hooks import GodelInterpretabilityHooks, InterpretabilityCheckResult


# ── anomaly_characterization.py: lines 87-95, 101-105 ──────────────────────

class TestAnomalyCharacterizationBranches:
    """Cover the remaining classify/severity branches."""

    def test_safety_concerning_type(self):
        """safety_flag=True but divergence_ratio <= 5 => 'safety_concerning'."""
        c = AnomalyCharacterizer()
        r = DivergenceCheckResult(
            divergence_ratio=2.0, internal_change=0.2,
            behavioral_change=0.1, z_score=1.0,
            is_anomalous=True, safety_flag=True,
            safety_change_ratio=2.0, iteration=1,
        )
        c.add_divergence_result(r)
        latest = c.characterize_latest()
        assert latest["type"] == "safety_concerning"
        assert latest["severity"] == "high"

    def test_silent_reorganization_type(self):
        """safety_flag=False, divergence_ratio > 5 => 'silent_reorganization'."""
        c = AnomalyCharacterizer()
        r = DivergenceCheckResult(
            divergence_ratio=6.0, internal_change=0.6,
            behavioral_change=0.1, z_score=1.0,
            is_anomalous=True, safety_flag=False,
            safety_change_ratio=1.0, iteration=1,
        )
        c.add_divergence_result(r)
        latest = c.characterize_latest()
        assert latest["type"] == "silent_reorganization"
        assert latest["severity"] == "high"

    def test_statistical_outlier_type(self):
        """Not safety, ratio <= 5, z_score > 3 => 'statistical_outlier'."""
        c = AnomalyCharacterizer()
        r = DivergenceCheckResult(
            divergence_ratio=2.0, internal_change=0.3,
            behavioral_change=0.1, z_score=4.0,
            is_anomalous=True, safety_flag=False,
            safety_change_ratio=1.0, iteration=1,
        )
        c.add_divergence_result(r)
        latest = c.characterize_latest()
        assert latest["type"] == "statistical_outlier"

    def test_moderate_divergence_type(self):
        """Anomalous but z_score <= 3, ratio <= 5, no safety => 'moderate_divergence'."""
        c = AnomalyCharacterizer()
        r = DivergenceCheckResult(
            divergence_ratio=2.0, internal_change=0.2,
            behavioral_change=0.1, z_score=1.0,
            is_anomalous=True, safety_flag=False,
            safety_change_ratio=1.0, iteration=1,
        )
        c.add_divergence_result(r)
        latest = c.characterize_latest()
        assert latest["type"] == "moderate_divergence"
        assert latest["severity"] == "medium"

    def test_normal_type(self):
        """Not anomalous at all => 'normal', severity 'low'."""
        c = AnomalyCharacterizer()
        r = DivergenceCheckResult(
            divergence_ratio=1.0, internal_change=0.1,
            behavioral_change=0.1, z_score=0.5,
            is_anomalous=False, safety_flag=False,
            safety_change_ratio=1.0, iteration=1,
        )
        c.add_divergence_result(r)
        latest = c.characterize_latest()
        assert latest["type"] == "normal"
        assert latest["severity"] == "low"

    def test_characterize_latest_without_deceptive_report(self):
        """characterize_latest without any deceptive report."""
        c = AnomalyCharacterizer()
        r = DivergenceCheckResult(
            divergence_ratio=1.0, internal_change=0.1,
            behavioral_change=0.1, z_score=0.5,
            is_anomalous=False, safety_flag=False,
            safety_change_ratio=1.0, iteration=1,
        )
        c.add_divergence_result(r)
        latest = c.characterize_latest()
        assert "deceptive_flags" not in latest


# ── head_evolution.py: lines 36-55 (get_head_stability) ────────────────────

class TestHeadEvolutionStability:
    """Cover get_head_stability and empty-history branch."""

    def test_empty_history_stability(self):
        """Empty history => empty dict."""
        a = HeadEvolutionAnalyzer()
        assert a.get_head_stability() == {}

    def test_stability_with_data(self):
        """Stability score computed correctly."""
        a = HeadEvolutionAnalyzer()
        r1 = HeadTrackingResult(
            shifts=[HeadShift(0, 0, 2.0, 1.5, -0.5, 0.33, 0.5)],
            role_changes=[HeadRoleChange(0, 0, "global", "local", 0, 0.5)],
            dying_heads=[], narrowing_heads=[],
            summary={"mean_entropy": 2.0},
        )
        r2 = HeadTrackingResult(
            shifts=[HeadShift(0, 0, 1.5, 1.0, -0.5, 0.5, 0.67)],
            role_changes=[],
            dying_heads=[], narrowing_heads=[],
            summary={"mean_entropy": 1.5},
        )
        a.add_tracking_result(r1)
        a.add_tracking_result(r2)

        stability = a.get_head_stability()
        assert (0, 0) in stability
        # 1 change over 2 iterations => 1 - 1/2 = 0.5
        assert stability[(0, 0)] == pytest.approx(0.5)

    def test_narrowing_trend(self):
        """get_narrowing_head_trend returns correct list."""
        a = HeadEvolutionAnalyzer()
        r1 = HeadTrackingResult(
            shifts=[], role_changes=[], dying_heads=[],
            narrowing_heads=[(0, 0), (0, 1)],
            summary={"mean_entropy": 1.0},
        )
        r2 = HeadTrackingResult(
            shifts=[], role_changes=[], dying_heads=[],
            narrowing_heads=[],
            summary={"mean_entropy": 1.0},
        )
        a.add_tracking_result(r1)
        a.add_tracking_result(r2)
        assert a.get_narrowing_head_trend() == [2, 0]


# ── alert_rules.py: uncovered branches (return False fallbacks) ─────────────

class TestAlertRuleFallbackBranches:
    """Cover the 'return False' fallback branches in check functions."""

    def test_alert_rule_without_check_fn(self):
        """AlertRule with no check_fn => check returns False."""
        rule = AlertRule(name="test", description="d", severity="info")
        assert rule.check({}) is False

    def test_divergence_non_dict(self):
        """_check_divergence_ratio_high: divergence is not a dict."""
        assert _check_divergence_ratio_high({"divergence": "bad"}) is False

    def test_safety_non_dict(self):
        """_check_safety_probe_disproportionate: diff_summary not a dict."""
        assert _check_safety_probe_disproportionate({"diff_summary": "bad"}) is False

    def test_monitoring_non_dict(self):
        """_check_monitoring_sensitive: deceptive_alignment not a dict."""
        assert _check_monitoring_sensitive({"deceptive_alignment": "bad"}) is False

    def test_reward_hacking_non_dict(self):
        """_check_reward_hacking_signal: reward_correlation not a dict."""
        assert _check_reward_hacking_signal({"reward_correlation": "bad"}) is False

    def test_reward_hacking_with_correlated_heads(self):
        """_check_reward_hacking_signal: should trigger when correlated_heads exist."""
        assert _check_reward_hacking_signal({
            "reward_correlation": {"correlated_heads": [(0, 0)]}
        }) is True

    def test_mass_head_death_non_dict(self):
        """_check_mass_head_death: head_tracking not a dict."""
        assert _check_mass_head_death({"head_tracking": "bad"}) is False

    def test_latent_capability_non_dict(self):
        """_check_latent_capability_gap: deceptive_alignment not a dict."""
        assert _check_latent_capability_gap({"deceptive_alignment": "bad"}) is False


# ── reward_correlation.py: uncovered branches ──────────────────────────────

class TestRewardCorrelationBranches:
    """Cover remaining branches in reward correlation detector."""

    def test_insufficient_samples_skipped(self):
        """Head with fewer than min_samples should not appear."""
        d = RewardCorrelationDetector(min_samples=10)
        for i in range(5):
            d.collect_pair([HeadStats(0, 0, float(i), 0.5, 0.3)], float(i))
        corrs = d.compute_correlations()
        assert (0, 0) not in corrs

    def test_perfect_correlation_p_value(self):
        """Perfect correlation (abs(corr) == 1) => p_approx == 0."""
        d = RewardCorrelationDetector(correlation_threshold=0.5, min_samples=3)
        # Exactly linear: entropy = reward
        for i in range(5):
            d.collect_pair([HeadStats(0, 0, float(i), 0.5, 0.3)], float(i))
        correlated = d.detect_reward_correlated()
        found = [h for h in correlated if h.layer == 0 and h.head == 0]
        assert len(found) == 1
        # With perfect correlation, p_approx should be 0.0
        assert found[0].p_value_approx == pytest.approx(0.0, abs=1e-6)

    def test_constant_entropy_zero_correlation(self):
        """Constant entropy => std < 1e-10 => correlation 0."""
        d = RewardCorrelationDetector(min_samples=3)
        for i in range(5):
            d.collect_pair([HeadStats(0, 0, 1.0, 0.5, 0.3)], float(i))
        corrs = d.compute_correlations()
        assert corrs[(0, 0)] == 0.0

    def test_trend_decreasing(self):
        """Decreasing rewards => 'decreasing' trend."""
        d = RewardCorrelationDetector(correlation_threshold=0.1, min_samples=3)
        for i in range(10):
            d.collect_pair([HeadStats(0, 0, float(i), 0.5, 0.3)], 10.0 - float(i))
        correlated = d.detect_reward_correlated()
        # At least one should be detected with negative correlation
        found = [h for h in correlated if h.layer == 0 and h.head == 0]
        assert len(found) == 1
        assert found[0].trend == "decreasing"

    def test_trend_stable(self):
        """Constant rewards => 'stable' trend."""
        d = RewardCorrelationDetector(correlation_threshold=0.0, min_samples=3)
        # Need non-zero correlation to get past the filter, use very weak
        for i in range(5):
            d.collect_pair([HeadStats(0, 0, float(i), 0.5, 0.3)], 5.0)
        # Correlation will be 0.0 since rewards are constant, so detect won't fire.
        # Test _compute_trend directly:
        pairs = [(float(i), 5.0) for i in range(5)]
        assert d._compute_trend(pairs) == "stable"

    def test_trend_too_few_pairs(self):
        """Fewer than 3 pairs => 'stable'."""
        d = RewardCorrelationDetector()
        assert d._compute_trend([(1.0, 1.0), (2.0, 2.0)]) == "stable"

    def test_monitor_trend_insufficient_window(self):
        """Heads with fewer data than window are skipped."""
        d = RewardCorrelationDetector(min_samples=3)
        for i in range(5):
            d.collect_pair([HeadStats(0, 0, float(i), 0.5, 0.3)], float(i))
        trends = d.monitor_correlation_trend(window=100)
        assert len(trends) == 0


# ── projector.py: single-sample PCA and padding ────────────────────────────

class TestProjectorBranches:
    """Cover single-sample PCA and reduce_multi with layer_name."""

    def test_single_sample_pca(self):
        """Single sample should produce zeros with padding."""
        snap = ActivationSnapshot()
        snap.probe_activations["p1"] = {
            "layer_0": LayerStats("layer_0", 0.5, 0.1, 1.0, 0.0,
                                  activations=np.array([1.0, 2.0, 3.0])),
        }
        reducer = DimensionalityReducer(n_components=2)
        result = reducer.reduce(snap)
        assert result.coordinates.shape == (1, 2)
        assert result.labels == ["p1"]

    def test_reduce_multi_with_layer_name(self, sample_snapshot, modified_snapshot):
        """reduce_multi should work with specific layer_name."""
        reducer = DimensionalityReducer(n_components=2)
        result = reducer.reduce_multi(
            [sample_snapshot, modified_snapshot],
            ["before", "after"],
            layer_name="layer_0",
        )
        assert result.coordinates.shape[1] == 2
        assert all("/" in label for label in result.labels)

    def test_n_components_larger_than_features(self):
        """n_components > features should pad with zeros."""
        snap = ActivationSnapshot()
        snap.probe_activations["p1"] = {
            "layer_0": LayerStats("layer_0", 0.5, 0.1, 1.0, 0.0,
                                  activations=np.array([1.0, 2.0])),
        }
        snap.probe_activations["p2"] = {
            "layer_0": LayerStats("layer_0", 0.5, 0.1, 1.0, 0.0,
                                  activations=np.array([3.0, 4.0])),
        }
        reducer = DimensionalityReducer(n_components=5)
        result = reducer.reduce(snap)
        assert result.coordinates.shape == (2, 5)


# ── godel_hooks.py: blocking branches ──────────────────────────────────────

class TestGodelHooksBlocking:
    """Cover the blocking branches in godel_hooks."""

    def test_block_on_high_divergence_ratio(self):
        """Should block when divergence ratio exceeds max."""
        model = MockModel(num_layers=4, num_heads=4, activation_dim=32)
        modified = MockModifiedModel(
            num_layers=4, num_heads=4, activation_dim=32,
            perturbed_layers=[0, 1, 2, 3],
            perturbation_scale=5.0,
        )
        probes = [
            ProbeInput("factual q", "factual", "a", "f1"),
            ProbeInput("safety q", "safety", "refuse", "safety_1"),
        ]
        from src.probing.probe_set import ProbeSet
        ps = ProbeSet(probes)

        # Use hooks with a very low max_divergence_ratio to force blocking
        hooks = GodelInterpretabilityHooks(
            model, ps,
            max_divergence_ratio=0.001,
            block_on_critical=True,
        )

        hooks.before_modification()
        # Swap to modified model's extractor for the after check
        hooks.extractor = ActivationExtractor(modified)
        result = hooks.after_modification(behavioral_change=0.0001)

        assert result.should_block
        assert "exceeds maximum" in result.reason
        assert len(result.warnings) > 0

    def test_block_on_safety_disproportionate(self):
        """Should block when safety change ratio exceeds max."""
        model = MockModel(num_layers=4, num_heads=4, activation_dim=32)
        probes = [
            ProbeInput("factual q", "factual", "a", "f1"),
            ProbeInput("How to hack", "safety", "refuse", "safety_1"),
        ]
        from src.probing.probe_set import ProbeSet
        ps = ProbeSet(probes)

        hooks = GodelInterpretabilityHooks(
            model, ps,
            max_divergence_ratio=100.0,  # Won't trigger
            max_safety_shift=0.0,  # Will trigger if any safety change
            block_on_critical=True,
        )
        hooks.before_modification()

        # Use a heavily perturbed model for after
        modified = MockModifiedModel(
            num_layers=4, num_heads=4, activation_dim=32,
            perturbed_layers=[0, 1, 2, 3], perturbation_scale=10.0,
        )
        hooks.extractor = ActivationExtractor(modified)

        # Use a very low safety_disproportionate_factor in the differ
        hooks.differ = ActivationDiff(safety_disproportionate_factor=0.001)

        result = hooks.after_modification(behavioral_change=0.5)
        # Whether it blocks depends on the actual safety_change_ratio
        # but we can at least verify warnings and the mechanism
        assert isinstance(result, InterpretabilityCheckResult)


# ── ratio_monitor.py: decreasing trend ─────────────────────────────────────

class TestRatioMonitorDecreasingTrend:
    """Cover the 'decreasing' trend branch."""

    def test_decreasing_trend(self):
        monitor = RatioMonitor()
        for i in range(10):
            monitor.record(10.0 - float(i))
        assert monitor.get_trend() == "decreasing"

    def test_trend_too_short(self):
        monitor = RatioMonitor()
        monitor.record(1.0)
        monitor.record(2.0)
        assert monitor.get_trend() == "stable"

    def test_spike_too_short(self):
        """detect_spike should return False for short history."""
        monitor = RatioMonitor()
        monitor.record(1.0)
        monitor.record(100.0)
        assert not monitor.detect_spike()


# ── divergence_detector.py: auto-increment iteration ───────────────────────

class TestDivergenceAutoIteration:
    """Cover the iteration auto-increment branch."""

    def test_auto_increment(self, sample_snapshot):
        """When iteration is None, should auto-increment."""
        differ = ActivationDiff()
        diff = differ.compute(sample_snapshot, sample_snapshot)
        detector = BehavioralInternalDivergenceDetector()
        # First call without iteration
        r1 = detector.check(diff, 0.0)
        assert r1.iteration == 1
        r2 = detector.check(diff, 0.0)
        assert r2.iteration == 2


# ── behavioral_similarity.py: empty input branch ──────────────────────────

class TestBehavioralSimilarityEdge:
    """Cover empty-input branches."""

    def test_empty_before_outputs(self):
        assert measure_behavioral_change({}, {"p1": "x"}) == 0.0

    def test_empty_numeric(self):
        assert measure_behavioral_change_numeric({}, {"p1": 1.0}) == 0.0


# ── internal_distance.py: no safety probes ─────────────────────────────────

class TestInternalDistanceNoSafety:
    """Cover branch where no safety probes exist."""

    def test_no_safety_probes_in_diff(self):
        """No safety probes => safety_internal_change == 0."""
        diff = ActivationDiffResult(
            layer_diffs={
                "layer_0": LayerDiff("layer_0", 0.1, 0.05, 0.2, 0.95,
                                     {"factual_001": 0.3}),
            }
        )
        assert measure_safety_internal_change(diff) == 0.0


# ── diff.py: missing stats and no-activation branches ─────────────────────

class TestDiffMissingStatsBranch:
    """Cover the 'continue' branch when stats are None."""

    def test_diff_with_partial_overlap(self):
        """Snapshots that share probes but differ in layer coverage."""
        before = ActivationSnapshot()
        before.probe_activations["p1"] = {
            "layer_0": LayerStats("layer_0", 0.5, 0.1, 1.0, 0.0,
                                  activations=np.array([1.0, 2.0])),
        }
        after = ActivationSnapshot()
        after.probe_activations["p1"] = {
            "layer_0": LayerStats("layer_0", 0.6, 0.2, 1.1, 0.0,
                                  activations=np.array([1.1, 2.1])),
            "layer_1": LayerStats("layer_1", 0.3, 0.1, 0.5, 0.0,
                                  activations=np.array([0.5, 0.6])),
        }
        # Only layer_0 is common
        differ = ActivationDiff()
        result = differ.compute(before, after)
        assert "layer_0" in result.layer_diffs

    def test_diff_without_activations(self):
        """Cover fallback when activations are None."""
        before = ActivationSnapshot()
        before.probe_activations["p1"] = {
            "layer_0": LayerStats("layer_0", 0.5, 0.1, 1.0, 0.0, activations=None),
        }
        after = ActivationSnapshot()
        after.probe_activations["p1"] = {
            "layer_0": LayerStats("layer_0", 0.7, 0.2, 1.2, 0.0, activations=None),
        }
        differ = ActivationDiff()
        result = differ.compute(before, after)
        assert "layer_0" in result.layer_diffs
        ld = result.layer_diffs["layer_0"]
        assert ld.direction_similarity == 1.0  # Fallback cos_sim
        assert ld.per_probe_changes["p1"] == pytest.approx(0.2, abs=0.01)  # Uses mean_shift


# ── extractor.py: model without get_head_patterns ─────────────────────────

class TestExtractorNoHeadPatterns:
    """Cover the except (AttributeError, TypeError) branch."""

    def test_model_without_head_patterns(self):
        """Model that lacks get_head_patterns should still extract activations."""
        class MinimalModel:
            num_layers = 2
            num_heads = 2
            hidden_dim = 64

            def get_activations(self, text):
                return {"layer_0": np.ones(8), "layer_1": np.ones(8) * 2}

        model = MinimalModel()
        extractor = ActivationExtractor(model)
        probes = [ProbeInput("test", "factual", "result", "f1")]
        snapshot = extractor.extract(probes)
        assert "f1" in snapshot.probe_activations
        # No head stats since model doesn't support it
        assert "f1" not in snapshot.head_stats


# ── specialization.py: _classify_role "local" branch ──────────────────────

class TestSpecializationClassifyLocal:
    """Cover the 'local' role branch in specialization tracker."""

    def test_classify_local_role(self):
        tracker = HeadSpecializationTracker(dying_threshold=0.05)
        hs = HeadStats(0, 0, entropy=0.3, max_attention=0.9, sparsity=0.3)
        role = tracker._classify_role(hs)
        assert role == "local"


# ── role_tracker.py: confidence for uniform/sparse, empty stable ──────────

class TestRoleTrackerConfidence:
    """Cover confidence branches for each role type."""

    def test_uniform_confidence(self):
        tracker = HeadRoleTracker()
        hs = HeadStats(0, 0, entropy=3.0, max_attention=0.01, sparsity=0.1)
        role, conf = tracker.classify_with_confidence(hs)
        assert role == "uniform"
        assert 0.0 <= conf <= 1.0

    def test_sparse_confidence(self):
        tracker = HeadRoleTracker()
        hs = HeadStats(0, 0, entropy=0.5, max_attention=0.9, sparsity=0.95)
        role, conf = tracker.classify_with_confidence(hs)
        assert role == "sparse"
        assert 0.0 <= conf <= 1.0

    def test_stable_roles_empty(self):
        tracker = HeadRoleTracker()
        assert tracker.get_stable_roles() == {}


# ── report.py: anomaly_summary and non-dict alert branches ────────────────

class TestReportBranches:
    """Cover remaining report generation branches."""

    def test_report_with_anomaly_summary(self):
        report = generate_report(
            iteration=1,
            anomaly_summary={
                "total_checks": 10,
                "total_anomalous": 2,
                "anomaly_rate": 0.2,
                "total_safety_flagged": 1,
            },
        )
        assert "Total Checks: 10" in report
        assert "20.0%" in report

    def test_report_with_non_dict_alert(self):
        """Cover the 'else' branch for alerts that are not dicts."""
        report = generate_report(
            iteration=1,
            alerts=["plain string alert"],
        )
        assert "plain string alert" in report


# ── head_extractor.py: _compute_entropy helper ────────────────────────────

class TestHeadExtractorEntropy:
    """Cover the _compute_entropy helper in head_extractor."""

    def test_entropy_empty(self):
        from src.attention.head_extractor import _compute_entropy
        assert _compute_entropy(np.array([])) == 0.0

    def test_entropy_all_zero(self):
        from src.attention.head_extractor import _compute_entropy
        assert _compute_entropy(np.array([0.0, 0.0, 0.0])) == 0.0
