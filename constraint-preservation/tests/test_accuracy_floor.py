"""Tests for AccuracyFloorConstraint."""

import pytest
from tests.conftest import MockAgent
from src.constraints.accuracy_floor import AccuracyFloorConstraint
from src.constraints.base import CheckContext


class TestAccuracyFloor:
    """AccuracyFloorConstraint test suite."""

    def test_pass_at_085(self, check_context):
        """Agent with 0.85 accuracy passes the 0.80 floor."""
        agent = MockAgent(accuracy=0.85)
        constraint = AccuracyFloorConstraint(threshold=0.80)
        result = constraint.check(agent, check_context)

        assert result.satisfied is True
        assert result.measured_value >= 0.80
        assert result.headroom > 0

    def test_fail_at_079(self, check_context):
        """Agent with 0.79 accuracy fails the 0.80 floor."""
        # Use a very low accuracy to ensure stochastic process yields < 0.80
        agent = MockAgent(accuracy=0.50)
        constraint = AccuracyFloorConstraint(threshold=0.80)
        result = constraint.check(agent, check_context)

        assert result.satisfied is False
        assert result.measured_value < 0.80
        assert result.headroom < 0

    def test_exact_boundary(self, check_context):
        """Accuracy exactly at threshold should pass."""
        constraint = AccuracyFloorConstraint(threshold=0.80)

        # Create a mock that returns exactly 80% correct
        class _ExactAgent:
            held_out_tasks = [{"category": "test", "prompt": f"q{i}"} for i in range(10)]

            def evaluate(self, tasks):
                results = []
                for i, t in enumerate(tasks):
                    results.append({"correct": i < 8, "category": t.get("category", "test")})
                return results

        agent = _ExactAgent()
        result = constraint.check(agent, check_context)

        assert result.satisfied is True
        assert abs(result.measured_value - 0.80) < 1e-9
        assert abs(result.headroom) < 1e-9

    def test_per_category_breakdown(self, check_context):
        """Result should include per-category accuracy."""
        agent = MockAgent(accuracy=0.85)
        constraint = AccuracyFloorConstraint(threshold=0.80)
        result = constraint.check(agent, check_context)

        assert "per_category" in result.details
        per_cat = result.details["per_category"]
        assert len(per_cat) > 0
        for cat, acc in per_cat.items():
            assert 0.0 <= acc <= 1.0

    def test_details_contain_totals(self, check_context):
        """Result details should contain total and correct counts."""
        agent = MockAgent(accuracy=0.90)
        constraint = AccuracyFloorConstraint(threshold=0.80)
        result = constraint.check(agent, check_context)

        assert "total" in result.details
        assert "correct" in result.details
        assert result.details["total"] > 0
        assert result.details["correct"] <= result.details["total"]

    def test_custom_threshold(self, check_context):
        """Custom threshold is respected."""
        constraint = AccuracyFloorConstraint(threshold=0.95)
        assert constraint.threshold == 0.95

        agent = MockAgent(accuracy=0.85)
        result = constraint.check(agent, check_context)
        # 0.85 accuracy may or may not hit 0.95 threshold -- just check structure
        assert result.threshold == 0.95

    def test_properties(self):
        """Constraint properties are correct."""
        constraint = AccuracyFloorConstraint(threshold=0.80)
        assert constraint.name == "accuracy_floor"
        assert constraint.category == "quality"
        assert constraint.threshold == 0.80
        assert constraint.is_immutable is True

    def test_headroom_computation(self, check_context):
        """Headroom should be measured - threshold."""
        agent = MockAgent(accuracy=0.90)
        constraint = AccuracyFloorConstraint(threshold=0.80)
        result = constraint.check(agent, check_context)

        expected_headroom = result.measured_value - 0.80
        assert abs(result.headroom - expected_headroom) < 1e-9
