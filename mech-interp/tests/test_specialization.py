"""Tests for attention head specialization tracking."""

import numpy as np
import pytest

from src.probing.extractor import HeadStats
from src.attention.specialization import (
    HeadSpecializationTracker, HeadShift, HeadRoleChange,
    HeadTrackingResult, measure_specialization,
)
from src.attention.head_extractor import HeadExtractor
from src.attention.dead_head_detector import DeadHeadDetector, DeadHead
from src.attention.role_tracker import HeadRoleTracker
from src.attention.reward_correlation import RewardCorrelationDetector, RewardCorrelatedHead


class TestMeasureSpecialization:
    """Test specialization measurement."""

    def test_zero_entropy_max_specialization(self):
        """Zero entropy should give max specialization."""
        assert measure_specialization(0.0) == 1.0

    def test_max_entropy_zero_specialization(self):
        """Max entropy should give zero specialization."""
        assert measure_specialization(3.0, max_entropy=3.0) == pytest.approx(0.0)

    def test_intermediate(self):
        """Intermediate entropy should give intermediate specialization."""
        spec = measure_specialization(1.5, max_entropy=3.0)
        assert 0.0 < spec < 1.0
        assert spec == pytest.approx(0.5)


class TestHeadSpecializationTracker:
    """Test head specialization tracking."""

    def test_uniform_attention_dying(self):
        """Heads with very low max attention should be detected as dying."""
        tracker = HeadSpecializationTracker(dying_threshold=0.05)
        stats = [
            HeadStats(layer=0, head=0, entropy=3.0, max_attention=0.02, sparsity=0.1),
            HeadStats(layer=0, head=1, entropy=1.0, max_attention=0.8, sparsity=0.5),
        ]
        result = tracker.track(stats)
        assert (0, 0) in result.dying_heads
        assert (0, 1) not in result.dying_heads

    def test_concentrated_specialized(self):
        """Concentrated attention should have high specialization."""
        stats_before = [
            HeadStats(layer=0, head=0, entropy=2.5, max_attention=0.3, sparsity=0.2),
        ]
        stats_after = [
            HeadStats(layer=0, head=0, entropy=0.5, max_attention=0.9, sparsity=0.8),
        ]
        tracker = HeadSpecializationTracker()
        tracker.track(stats_before)
        result = tracker.track(stats_after)

        # Should detect entropy drop (narrowing)
        assert len(result.shifts) > 0
        shift = result.shifts[0]
        assert shift.entropy_change < 0  # Entropy decreased

    def test_role_changes_detected(self):
        """Should detect when heads change role."""
        tracker = HeadSpecializationTracker(dying_threshold=0.05)

        # First iteration: head is "global"
        stats1 = [
            HeadStats(layer=0, head=0, entropy=2.0, max_attention=0.5, sparsity=0.3),
        ]
        tracker.track(stats1)

        # Second iteration: head becomes "sparse"
        stats2 = [
            HeadStats(layer=0, head=0, entropy=0.3, max_attention=0.95, sparsity=0.9),
        ]
        result = tracker.track(stats2)
        assert len(result.role_changes) > 0
        rc = result.role_changes[0]
        assert rc.role_before == "global"
        assert rc.role_after == "sparse"

    def test_narrowing_heads(self):
        """Should detect narrowing heads (large entropy drop)."""
        tracker = HeadSpecializationTracker(narrowing_entropy_drop=0.3)

        stats1 = [
            HeadStats(layer=0, head=0, entropy=2.0, max_attention=0.5, sparsity=0.3),
        ]
        tracker.track(stats1)

        stats2 = [
            HeadStats(layer=0, head=0, entropy=1.0, max_attention=0.7, sparsity=0.5),
        ]
        result = tracker.track(stats2)
        assert (0, 0) in result.narrowing_heads

    def test_summary_stats(self):
        """Should compute summary statistics."""
        tracker = HeadSpecializationTracker()
        stats = [
            HeadStats(layer=0, head=0, entropy=1.0, max_attention=0.5, sparsity=0.3),
            HeadStats(layer=0, head=1, entropy=2.0, max_attention=0.4, sparsity=0.2),
        ]
        result = tracker.track(stats)
        assert "mean_entropy" in result.summary
        assert result.summary["mean_entropy"] == pytest.approx(1.5)
        assert result.summary["num_heads"] == 2

    def test_detect_role_changes_across_history(self):
        """detect_role_changes should return all historical changes."""
        tracker = HeadSpecializationTracker(dying_threshold=0.05)
        # global -> sparse -> uniform
        tracker.track([HeadStats(0, 0, 2.0, 0.5, 0.3)])
        tracker.track([HeadStats(0, 0, 0.3, 0.95, 0.9)])
        tracker.track([HeadStats(0, 0, 3.0, 0.01, 0.1)])

        changes = tracker.detect_role_changes()
        assert len(changes) >= 2

    def test_tracking_result_to_dict(self):
        """HeadTrackingResult should serialize."""
        result = HeadTrackingResult(
            dying_heads=[(0, 1)],
            narrowing_heads=[(1, 2)],
            summary={"mean_entropy": 1.5},
        )
        d = result.to_dict()
        assert d["num_dying_heads"] == 1
        assert d["dying_heads"] == [[0, 1]]


class TestDeadHeadDetector:
    """Test dead head detection."""

    def test_detect_dead_heads(self):
        """Should detect heads with low max attention."""
        detector = DeadHeadDetector(max_attention_threshold=0.1)
        stats = [
            HeadStats(layer=0, head=0, entropy=3.0, max_attention=0.02, sparsity=0.1),
            HeadStats(layer=0, head=1, entropy=1.0, max_attention=0.8, sparsity=0.5),
            HeadStats(layer=1, head=0, entropy=2.5, max_attention=0.05, sparsity=0.2),
        ]
        dead = detector.detect(stats)
        assert len(dead) == 2
        dead_ids = [(d.layer, d.head) for d in dead]
        assert (0, 0) in dead_ids
        assert (1, 0) in dead_ids

    def test_mass_death_detection(self):
        """Should detect mass head death."""
        detector = DeadHeadDetector(max_attention_threshold=0.1)
        stats = [
            HeadStats(layer=0, head=0, entropy=3.0, max_attention=0.02, sparsity=0.1),
            HeadStats(layer=0, head=1, entropy=3.0, max_attention=0.03, sparsity=0.1),
            HeadStats(layer=0, head=2, entropy=1.0, max_attention=0.8, sparsity=0.5),
        ]
        assert detector.detect_mass_death(stats, threshold_fraction=0.5)

    def test_no_mass_death(self):
        """Should not flag when few heads are dead."""
        detector = DeadHeadDetector(max_attention_threshold=0.1)
        stats = [
            HeadStats(layer=0, head=0, entropy=1.0, max_attention=0.8, sparsity=0.5),
            HeadStats(layer=0, head=1, entropy=1.0, max_attention=0.7, sparsity=0.5),
            HeadStats(layer=0, head=2, entropy=1.0, max_attention=0.6, sparsity=0.5),
        ]
        assert not detector.detect_mass_death(stats)

    def test_get_dead_head_ids(self):
        """Should return (layer, head) tuples."""
        detector = DeadHeadDetector(max_attention_threshold=0.1)
        stats = [
            HeadStats(0, 0, 3.0, 0.02, 0.1),
            HeadStats(0, 1, 1.0, 0.8, 0.5),
        ]
        ids = detector.get_dead_head_ids(stats)
        assert (0, 0) in ids
        assert (0, 1) not in ids

    def test_entropy_threshold(self):
        """Should use entropy threshold when configured."""
        detector = DeadHeadDetector(max_attention_threshold=0.01, entropy_threshold=2.0)
        stats = [
            HeadStats(0, 0, 2.5, 0.5, 0.3),  # Only high entropy
        ]
        dead = detector.detect(stats)
        assert len(dead) == 1

    def test_empty_input(self):
        """Should handle empty input."""
        detector = DeadHeadDetector()
        assert not detector.detect_mass_death([])


class TestHeadRoleTracker:
    """Test head role tracking."""

    def test_classify_roles(self):
        """Should classify head roles correctly."""
        tracker = HeadRoleTracker()
        assert tracker.classify_role(HeadStats(0, 0, 3.0, 0.02, 0.1)) == "uniform"
        assert tracker.classify_role(HeadStats(0, 0, 0.3, 0.9, 0.9)) == "sparse"
        assert tracker.classify_role(HeadStats(0, 0, 0.2, 0.9, 0.3)) == "local"
        assert tracker.classify_role(HeadStats(0, 0, 2.0, 0.5, 0.3)) == "global"

    def test_track_roles_over_time(self):
        """Should track roles over iterations."""
        tracker = HeadRoleTracker()
        stats1 = [HeadStats(0, 0, 2.0, 0.5, 0.3)]
        stats2 = [HeadStats(0, 0, 0.2, 0.9, 0.3)]

        tracker.track(stats1)
        tracker.track(stats2)

        history = tracker.get_role_history(0, 0)
        assert history == ["global", "local"]

    def test_get_role_changes(self):
        """Should detect role changes."""
        tracker = HeadRoleTracker()
        tracker.track([HeadStats(0, 0, 2.0, 0.5, 0.3)])
        tracker.track([HeadStats(0, 0, 0.2, 0.9, 0.3)])

        changes = tracker.get_role_changes()
        assert len(changes) == 1
        assert changes[0]["from"] == "global"
        assert changes[0]["to"] == "local"

    def test_classify_with_confidence(self):
        """Should return role with confidence."""
        tracker = HeadRoleTracker()
        role, conf = tracker.classify_with_confidence(HeadStats(0, 0, 2.0, 0.5, 0.3))
        assert role == "global"
        assert 0.0 <= conf <= 1.0

    def test_stable_roles(self):
        """Should identify stable roles."""
        tracker = HeadRoleTracker()
        for _ in range(3):
            tracker.track([HeadStats(0, 0, 2.0, 0.5, 0.3)])
        stable = tracker.get_stable_roles()
        assert (0, 0) in stable
        assert stable[(0, 0)] == "global"


class TestHeadExtractor:
    """Test head extraction from model."""

    def test_extract_head_patterns(self, mock_model, sample_probes):
        """Should extract head patterns for all probes."""
        extractor = HeadExtractor(mock_model)
        results = extractor.extract_head_patterns(sample_probes[:2])
        assert len(results) == 2
        for pid, stats in results.items():
            assert len(stats) > 0
            for hs in stats:
                assert isinstance(hs, HeadStats)

    def test_extract_aggregate_stats(self, mock_model, sample_probes):
        """Should aggregate stats across probes."""
        extractor = HeadExtractor(mock_model)
        agg = extractor.extract_aggregate_stats(sample_probes[:3])
        assert len(agg) > 0
        # Should have num_layers * num_heads entries
        assert len(agg) == mock_model.num_layers * mock_model.num_heads


class TestRewardCorrelationDetector:
    """Test reward correlation detection."""

    def test_detect_correlated_heads(self):
        """Should detect heads correlated with reward."""
        detector = RewardCorrelationDetector(correlation_threshold=0.3, min_samples=3)

        # Create correlated data: entropy increases with reward
        for i in range(10):
            reward = float(i) / 10
            stats = [
                HeadStats(0, 0, entropy=float(i) * 0.1, max_attention=0.5, sparsity=0.3),
                HeadStats(0, 1, entropy=1.0, max_attention=0.5, sparsity=0.3),  # Uncorrelated
            ]
            detector.collect_pair(stats, reward)

        correlated = detector.detect_reward_correlated()
        # Head (0,0) should be correlated
        correlated_ids = [(h.layer, h.head) for h in correlated]
        assert (0, 0) in correlated_ids

    def test_no_correlation(self):
        """Should not detect correlation when none exists."""
        detector = RewardCorrelationDetector(correlation_threshold=0.8, min_samples=3)
        # Constant entropy, varying reward
        for i in range(10):
            stats = [HeadStats(0, 0, 1.0, 0.5, 0.3)]
            detector.collect_pair(stats, float(i) / 10)

        correlated = detector.detect_reward_correlated()
        assert len(correlated) == 0

    def test_correlation_trend(self):
        """Should track correlation trends."""
        detector = RewardCorrelationDetector(min_samples=3)
        for i in range(15):
            stats = [HeadStats(0, 0, float(i) * 0.1, 0.5, 0.3)]
            detector.collect_pair(stats, float(i) * 0.2)

        trends = detector.monitor_correlation_trend(window=10)
        assert len(trends) > 0

    def test_compute_correlations(self):
        """Should compute correlations for all tracked heads."""
        detector = RewardCorrelationDetector(min_samples=3)
        for i in range(5):
            stats = [HeadStats(0, 0, float(i), 0.5, 0.3)]
            detector.collect_pair(stats, float(i))

        corrs = detector.compute_correlations()
        assert (0, 0) in corrs
        assert abs(corrs[(0, 0)]) > 0.5  # Strong correlation expected
