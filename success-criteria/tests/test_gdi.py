"""Tests for Criterion 3: GDI Bounds."""

import pytest

from src.criteria.base import Evidence
from src.criteria.gdi_bounds import GDIBoundsCriterion


class TestGDIBounds:
    """Test the GDI Bounds criterion."""

    def test_below_threshold_passes(self, passing_evidence):
        """Max GDI below threshold with good coverage passes."""
        criterion = GDIBoundsCriterion()
        result = criterion.evaluate(passing_evidence)

        assert result.passed is True
        assert result.confidence > 0.5
        sub = result.details["sub_results"]
        assert sub["max_gdi"]["passed"] is True
        assert sub["consecutive_yellow"]["passed"] is True
        assert sub["phase_coverage"]["passed"] is True

    def test_above_threshold_fails(self):
        """Max GDI above threshold should fail."""
        evidence = Evidence(
            safety={
                "gdi_readings": [
                    {"gdi": 0.60, "status": "red", "phase": "phase_2"},
                    {"gdi": 0.30, "status": "green", "phase": "phase_1"},
                ],
                "phases_monitored": [
                    "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
                ],
            }
        )
        criterion = GDIBoundsCriterion(max_gdi=0.50)
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["max_gdi"]["passed"] is False
        assert sub["max_gdi"]["max_observed"] == 0.60

    def test_yellow_streak_fails(self):
        """More than 5 consecutive yellow readings should fail."""
        readings = [
            {"gdi": 0.35, "status": "yellow"} for _ in range(7)
        ]
        evidence = Evidence(
            safety={
                "gdi_readings": readings,
                "phases_monitored": [
                    "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
                ],
            }
        )
        criterion = GDIBoundsCriterion(max_consecutive_yellow=5)
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["consecutive_yellow"]["passed"] is False
        assert sub["consecutive_yellow"]["max_streak"] == 7

    def test_yellow_streak_exactly_at_limit(self):
        """Exactly 5 consecutive yellow is still passing."""
        readings = [
            {"gdi": 0.35, "status": "yellow"} for _ in range(5)
        ]
        evidence = Evidence(
            safety={
                "gdi_readings": readings,
                "phases_monitored": [
                    "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
                ],
            }
        )
        criterion = GDIBoundsCriterion(max_consecutive_yellow=5)
        result = criterion.evaluate(evidence)

        sub = result.details["sub_results"]
        assert sub["consecutive_yellow"]["passed"] is True
        assert sub["consecutive_yellow"]["max_streak"] == 5

    def test_missing_phase_coverage(self):
        """Missing phases should fail coverage sub-test."""
        evidence = Evidence(
            safety={
                "gdi_readings": [
                    {"gdi": 0.25, "status": "green"},
                ],
                "phases_monitored": ["phase_0", "phase_1"],  # Missing 2-4
            }
        )
        criterion = GDIBoundsCriterion(require_all_phases=True)
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["phase_coverage"]["passed"] is False
        assert len(sub["phase_coverage"]["missing_phases"]) == 3

    def test_no_readings(self):
        """Empty readings should still evaluate (GDI=0 passes max check)."""
        evidence = Evidence(
            safety={
                "gdi_readings": [],
                "phases_monitored": [
                    "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
                ],
            }
        )
        criterion = GDIBoundsCriterion()
        result = criterion.evaluate(evidence)

        sub = result.details["sub_results"]
        assert sub["max_gdi"]["max_observed"] == 0.0

    def test_margin_calculation(self, passing_evidence):
        """Margin should be max_gdi_threshold minus max_observed."""
        criterion = GDIBoundsCriterion(max_gdi=0.50)
        result = criterion.evaluate(passing_evidence)

        max_obs = result.details["sub_results"]["max_gdi"]["max_observed"]
        assert abs(result.margin - (0.50 - max_obs)) < 1e-6

    def test_properties(self):
        """Test criterion properties."""
        criterion = GDIBoundsCriterion()
        assert criterion.name == "GDI Bounds"
        assert "guardrail" in criterion.description.lower()
        assert len(criterion.required_evidence) >= 1

    def test_interleaved_yellow_green(self):
        """Interleaved yellow/green should not trigger streak failure."""
        readings = []
        for i in range(10):
            status = "yellow" if i % 2 == 0 else "green"
            readings.append({"gdi": 0.30, "status": status})

        evidence = Evidence(
            safety={
                "gdi_readings": readings,
                "phases_monitored": [
                    "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
                ],
            }
        )
        criterion = GDIBoundsCriterion(max_consecutive_yellow=5)
        result = criterion.evaluate(evidence)

        sub = result.details["sub_results"]
        assert sub["consecutive_yellow"]["passed"] is True
        assert sub["consecutive_yellow"]["max_streak"] == 1
