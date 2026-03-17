"""Tests for phase gate safety package."""

import pytest

from src.deliverables.phase_gate import (
    PhaseGateSafetyPackage,
    SafetyPackage,
    PackageValidation,
)
from src.deliverables.gdi_package import GDISummary, package_gdi
from src.deliverables.constraint_package import ConstraintSummary, package_constraints
from src.deliverables.interp_package import InterpSummary, package_interp
from src.deliverables.reward_package import RewardSummary, package_reward
from src.deliverables.safety_report import generate_safety_report


class TestPhaseGateSafetyPackage:
    """Test safety package assembly and validation."""

    def test_all_green_package(self, green_histories):
        """All-green histories produce a valid, all-green package."""
        gate = PhaseGateSafetyPackage()
        pkg = gate.package("phase_1", (0, 100), green_histories)

        assert isinstance(pkg, SafetyPackage)
        assert pkg.all_green
        assert pkg.phase == "phase_1"
        assert pkg.iteration_range == (0, 100)
        assert pkg.gdi.is_green
        assert pkg.constraint.is_green
        assert pkg.interp.is_green
        assert pkg.reward.is_green

    def test_all_green_validation(self, green_histories):
        """All-green package passes validation."""
        gate = PhaseGateSafetyPackage()
        pkg = gate.package("phase_1", (0, 100), green_histories)
        validation = gate.validate(pkg)

        assert isinstance(validation, PackageValidation)
        assert validation.is_valid
        assert validation.errors == []
        assert validation.cross_signal_consistent

    def test_failing_package(self, failing_histories):
        """Failing histories produce a non-green package."""
        gate = PhaseGateSafetyPackage()
        pkg = gate.package("phase_2", (100, 200), failing_histories)

        assert not pkg.all_green
        assert pkg.gdi.status == "red"  # Has unresolved failures
        assert pkg.constraint.status == "red"
        assert pkg.interp.status == "red"
        assert pkg.reward.status == "red"

    def test_failing_validation(self, failing_histories):
        """Failing package does not pass validation."""
        gate = PhaseGateSafetyPackage()
        pkg = gate.package("phase_2", (100, 200), failing_histories)
        validation = gate.validate(pkg)

        assert not validation.is_valid
        assert len(validation.errors) > 0
        # Should mention unresolved failures
        assert any("unresolved" in e.lower() for e in validation.errors)

    def test_cross_signal_inconsistency(self):
        """Detects cross-signal inconsistency."""
        gate = PhaseGateSafetyPackage()

        # Reward hacking (red) but constraints all green -- inconsistent
        histories = {
            "gdi": {},
            "constraint": {
                "reward_bounded": True,
                "entropy_above_min": True,
                "energy_stable": True,
            },
            "interp": {},
            "reward": {
                "no_divergence": False,
                "no_shortcuts": False,
                "no_gaming": True,
            },
        }

        pkg = gate.package("phase_3", (200, 300), histories)
        validation = gate.validate(pkg)

        # Should detect cross-signal inconsistency
        assert not validation.cross_signal_consistent

    def test_partial_failures(self):
        """Package with some tracks yellow."""
        gate = PhaseGateSafetyPackage()

        histories = {
            "gdi": {},
            "constraint": {
                "reward_bounded": True,
                "entropy_above_min": False,  # Yellow
                "energy_stable": True,
            },
            "interp": {},
            "reward": {},
        }

        pkg = gate.package("phase_1", (0, 100), histories)
        assert not pkg.all_green
        assert pkg.constraint.status == "yellow"

    def test_packages_stored(self, green_histories):
        """Packages and validations are stored."""
        gate = PhaseGateSafetyPackage()
        pkg = gate.package("phase_1", (0, 100), green_histories)
        gate.validate(pkg)

        assert len(gate.packages) == 1
        assert len(gate.validations) == 1


class TestIndividualPackages:
    """Test individual track packaging functions."""

    def test_gdi_green(self):
        """Green GDI summary."""
        summary = package_gdi({})
        assert summary.is_green

    def test_gdi_with_issues(self):
        """GDI with unresolved failures is red."""
        summary = package_gdi({"unresolved_failures": ["critical_bug"]})
        assert summary.status == "red"

    def test_constraint_green(self):
        """Green constraint summary."""
        summary = package_constraints({
            "reward_bounded": True,
            "entropy_above_min": True,
            "energy_stable": True,
        })
        assert summary.is_green
        assert summary.constraints_met == 3

    def test_constraint_yellow(self):
        """Yellow constraint with one failure."""
        summary = package_constraints({
            "reward_bounded": True,
            "entropy_above_min": False,
            "energy_stable": True,
        })
        assert summary.status == "yellow"
        assert summary.constraints_met == 2

    def test_constraint_red(self):
        """Red constraint with multiple failures."""
        summary = package_constraints({
            "reward_bounded": False,
            "entropy_above_min": False,
            "energy_stable": True,
        })
        assert summary.status == "red"

    def test_interp_green(self):
        """Green interpretability summary."""
        summary = package_interp({})
        assert summary.is_green

    def test_interp_yellow(self):
        """Yellow with non-critical issue."""
        summary = package_interp({"energy_interpretable": False})
        assert summary.status == "yellow"

    def test_reward_green(self):
        """Green reward summary."""
        summary = package_reward({})
        assert summary.is_green

    def test_reward_yellow(self):
        """Yellow with single signal."""
        summary = package_reward({"no_divergence": False})
        assert summary.status == "yellow"
        assert "divergence" in summary.hacking_signals

    def test_reward_red(self):
        """Red with multiple signals."""
        summary = package_reward({"no_divergence": False, "no_shortcuts": False})
        assert summary.status == "red"


class TestSafetyReport:
    """Test safety report generation."""

    def test_report_all_green(self, green_histories):
        """Report for all-green package."""
        gate = PhaseGateSafetyPackage()
        pkg = gate.package("phase_1", (0, 100), green_histories)
        report = generate_safety_report(pkg)

        assert isinstance(report, str)
        assert "# Safety Report" in report
        assert "PASS" in report
        assert "Executive Summary" in report
        assert "GDI Track" in report
        assert "Constraint Track" in report
        assert "Interpretability Track" in report
        assert "Reward Integrity Track" in report
        assert "Cross-Signal Analysis" in report
        assert "Risk Assessment" in report
        assert "Recommendations" in report

    def test_report_failing(self, failing_histories):
        """Report for failing package."""
        gate = PhaseGateSafetyPackage()
        pkg = gate.package("phase_2", (100, 200), failing_histories)
        report = generate_safety_report(pkg)

        assert "FAIL" in report
        assert "RED" in report

    def test_report_has_8_sections(self, green_histories):
        """Report has all 8 required sections."""
        gate = PhaseGateSafetyPackage()
        pkg = gate.package("phase_1", (0, 100), green_histories)
        report = generate_safety_report(pkg)

        for i in range(1, 9):
            assert f"## {i}." in report
