"""Tests for EntropyFloorConstraint."""

import pytest
from tests.conftest import MockAgent
from src.constraints.entropy_floor import EntropyFloorConstraint
from src.constraints.base import CheckContext


class TestEntropyFloor:
    """EntropyFloorConstraint test suite."""

    def test_diverse_outputs_pass(self, check_context):
        """High-diversity agent passes the entropy floor."""
        agent = MockAgent(entropy=5.0)
        constraint = EntropyFloorConstraint(threshold=3.5)
        result = constraint.check(agent, check_context)

        assert result.satisfied is True
        assert result.measured_value >= 3.5
        assert result.headroom > 0

    def test_repetitive_outputs_fail(self, check_context):
        """Low-diversity agent fails the entropy floor."""
        agent = MockAgent(entropy=1.0)
        constraint = EntropyFloorConstraint(threshold=3.5)
        result = constraint.check(agent, check_context)

        assert result.satisfied is False
        assert result.measured_value < 3.5
        assert result.headroom < 0

    def test_boundary(self, check_context):
        """Test near the boundary."""
        # With entropy=3.0 the mock should produce moderate diversity
        agent = MockAgent(entropy=3.0)
        constraint = EntropyFloorConstraint(threshold=3.5)
        result = constraint.check(agent, check_context)

        # The measured entropy might be above or below, just verify structure
        assert isinstance(result.satisfied, bool)
        assert isinstance(result.measured_value, float)
        assert result.threshold == 3.5

    def test_details_metrics(self, check_context):
        """Details should contain all diversity metrics."""
        agent = MockAgent(entropy=5.0)
        constraint = EntropyFloorConstraint(threshold=3.5)
        result = constraint.check(agent, check_context)

        assert "token_entropy" in result.details
        assert "distinct_1" in result.details
        assert "distinct_2" in result.details
        assert "self_bleu" in result.details
        assert "vocab_size" in result.details

    def test_properties(self):
        """Constraint properties are correct."""
        constraint = EntropyFloorConstraint(threshold=4.0)
        assert constraint.name == "entropy_floor"
        assert constraint.category == "quality"
        assert constraint.threshold == 4.0
        assert constraint.is_immutable is True

    def test_very_high_entropy(self, check_context):
        """Very high entropy should easily pass."""
        agent = MockAgent(entropy=8.0)
        constraint = EntropyFloorConstraint(threshold=3.5)
        result = constraint.check(agent, check_context)

        assert result.satisfied is True
        assert result.headroom > 0

    def test_compute_diversity_metrics_empty(self):
        """Empty outputs should return zero entropy."""
        metrics = EntropyFloorConstraint._compute_diversity_metrics([])
        assert metrics["token_entropy"] == 0.0
        assert metrics["vocab_size"] == 0

    def test_compute_diversity_metrics_single_token(self):
        """Single repeated token has zero entropy."""
        metrics = EntropyFloorConstraint._compute_diversity_metrics(["a a a a a"])
        assert metrics["token_entropy"] == 0.0
        assert metrics["vocab_size"] == 1
