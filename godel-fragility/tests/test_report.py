"""Tests for the report generation module."""

from __future__ import annotations

import os

import pytest

from src.analysis.complexity_ceiling import CeilingAnalysis, SigmoidFit
from src.analysis.failure_landscape import FailureLandscape, Vulnerability
from src.analysis.fragility_score import FragilityReport
from src.analysis.report import (
    _complexity_ceiling_section,
    _detailed_results,
    _executive_summary,
    _failure_landscape_section,
    _fragility_score_section,
    _recovery_analysis_section,
    _stress_test_coverage,
    _vulnerabilities_section,
    generate_report,
)
from src.harness.stress_runner import ScenarioResult, StressTestResults
from src.measurement.failure_classifier import FailureMode
from src.measurement.recovery_tracker import RecoveryTracker


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


def _make_results() -> StressTestResults:
    """Build synthetic StressTestResults."""
    r1 = ScenarioResult(
        scenario_name="test_syntax",
        category="fault_injection",
        severity=3,
        success=True,
        failure_mode=FailureMode.VALIDATION_CAUGHT,
        accuracies=[0.8, 0.7, 0.75, 0.85],
        complexities=[30, 35, 38, 40],
        modification_count=4,
        rollback_count=1,
        iterations_run=4,
        duration_seconds=1.5,
    )
    r2 = ScenarioResult(
        scenario_name="test_logic",
        category="fault_injection",
        severity=4,
        success=False,
        failure_mode=FailureMode.SILENT_DEGRADATION,
        accuracies=[0.9, 0.85, 0.60, 0.40],
        complexities=[30, 32, 50, 65],
        modification_count=6,
        rollback_count=0,
        iterations_run=4,
        duration_seconds=2.0,
    )
    r3 = ScenarioResult(
        scenario_name="test_self_ref",
        category="self_reference",
        severity=5,
        success=False,
        failure_mode=FailureMode.SELF_LOBOTOMY,
        accuracies=[0.8, 0.5, 0.2],
        complexities=[30, 45, 60],
        modification_count=10,
        rollback_count=3,
        iterations_run=3,
        duration_seconds=3.0,
        timed_out=False,
    )
    return StressTestResults(
        results=[r1, r2, r3],
        total_scenarios=3,
        passed=1,
        failed=2,
        timed_out=0,
        duration_seconds=6.5,
    )


def _make_recovery_tracker() -> RecoveryTracker:
    tracker = RecoveryTracker()
    event = tracker.track_injection("test_syntax", "syntax", 0, complexity=30)
    tracker.update(1, 0.5, detected=True, detection_method="accuracy_drop")
    tracker.update(2, 0.85, recovered=True, recovery_method="rollback", recovery_quality=0.85)
    tracker.finalize_event()

    tracker.track_injection("test_logic", "logic", 0, complexity=30)
    tracker.update(1, 0.85)
    tracker.update(2, 0.60)
    tracker.update(3, 0.40)
    tracker.finalize_event()
    return tracker


def _make_ceiling_analysis() -> CeilingAnalysis:
    return CeilingAnalysis(
        ceiling_estimate=75.0,
        sigmoid_fit=SigmoidFit(L=1.0, k=-0.05, x0=75.0, b=0.0, r_squared=0.85),
        cliff_detected=True,
        cliff_location=70.0,
        decline_characterization="cliff",
        safe_operating_range=(10.0, 60.0),
        data_points=20,
        complexity_range=(10.0, 100.0),
    )


def _make_fragility_report() -> FragilityReport:
    return FragilityReport(
        overall_score=0.55,
        components={
            "recovery_rate": 0.50,
            "ceiling_ratio": 0.60,
            "catastrophic_rate": 0.33,
            "detection_rate": 0.50,
        },
        interpretation="The agent has notable fragility.",
        grade="C-",
        recommendations=[
            "Improve recovery mechanisms.",
            "Address low complexity ceiling.",
        ],
    )


def _make_landscape() -> FailureLandscape:
    return FailureLandscape(
        failure_mode_counts={"silent_degradation": 1, "self_lobotomy": 1},
        category_failure_rates={"fault_injection": 0.5, "self_reference": 1.0},
        severity_distribution={4: 1, 5: 1},
        vulnerabilities=[
            Vulnerability(
                name="self_lobotomy_vulnerability",
                category="self_reference",
                severity=5,
                description="Agent lobotomized itself in 1 scenario(s)",
                failure_mode=FailureMode.SELF_LOBOTOMY,
                affected_scenarios=["test_self_ref"],
                remediation="Add immutable core functions.",
            ),
        ],
        total_scenarios=3,
        total_failures=2,
    )


# ------------------------------------------------------------------ #
# generate_report
# ------------------------------------------------------------------ #


class TestGenerateReport:
    def test_minimal_report(self) -> None:
        results = _make_results()
        report = generate_report(results)
        assert "# Godel Agent Fragility Report" in report
        assert "## Executive Summary" in report
        assert "## Stress Test Coverage" in report
        assert "## Detailed Scenario Results" in report
        assert "Scenarios tested:** 3" in report
        assert "Passed:** 1" in report

    def test_full_report_all_sections(self) -> None:
        results = _make_results()
        tracker = _make_recovery_tracker()
        ceiling = _make_ceiling_analysis()
        fragility = _make_fragility_report()
        landscape = _make_landscape()

        report = generate_report(
            results,
            recovery_tracker=tracker,
            ceiling_analysis=ceiling,
            fragility_report=fragility,
            landscape=landscape,
        )
        assert "## Recovery Analysis" in report
        assert "## Complexity Ceiling" in report
        assert "## Fragility Score" in report
        assert "## Failure Landscape" in report
        assert "## Critical Vulnerabilities" in report

    def test_report_writes_to_file(self, tmp_path) -> None:
        results = _make_results()
        output_path = str(tmp_path / "report.md")
        report = generate_report(results, output_path=output_path)
        assert os.path.exists(output_path)
        with open(output_path) as f:
            content = f.read()
        assert content == report


# ------------------------------------------------------------------ #
# Individual section functions
# ------------------------------------------------------------------ #


class TestExecutiveSummary:
    def test_without_fragility(self) -> None:
        results = _make_results()
        section = _executive_summary(results, None)
        assert "Scenarios tested" in section
        assert "Passed" in section
        assert "Failed" in section
        assert "Duration" in section

    def test_with_fragility(self) -> None:
        results = _make_results()
        fragility = _make_fragility_report()
        section = _executive_summary(results, fragility)
        assert "Fragility score" in section
        assert "Grade" in section
        assert fragility.interpretation in section


class TestStressTestCoverage:
    def test_coverage_section(self) -> None:
        results = _make_results()
        section = _stress_test_coverage(results)
        assert "By Category" in section
        assert "Failure Mode Distribution" in section
        assert "fault_injection" in section
        assert "self_reference" in section


class TestFailureLandscapeSection:
    def test_landscape_section(self) -> None:
        landscape = _make_landscape()
        section = _failure_landscape_section(landscape)
        assert "Total failures: 2" in section
        assert "fault_injection" in section
        assert "self_reference" in section
        assert "Severity 4" in section
        assert "Severity 5" in section

    def test_empty_rates(self) -> None:
        landscape = FailureLandscape(total_scenarios=5, total_failures=0)
        section = _failure_landscape_section(landscape)
        assert "Total failures: 0" in section


class TestRecoveryAnalysisSection:
    def test_recovery_section(self) -> None:
        tracker = _make_recovery_tracker()
        section = _recovery_analysis_section(tracker)
        assert "Recovery rate" in section
        assert "Detection rate" in section
        assert "Recovery by Fault Type" in section


class TestComplexityCeilingSection:
    def test_with_full_analysis(self) -> None:
        analysis = _make_ceiling_analysis()
        section = _complexity_ceiling_section(analysis)
        assert "Estimated ceiling" in section
        assert "75" in section
        assert "cliff" in section.lower()
        assert "Safe operating range" in section
        assert "Sigmoid Fit" in section
        assert "R-squared" in section

    def test_without_ceiling(self) -> None:
        analysis = CeilingAnalysis()
        section = _complexity_ceiling_section(analysis)
        assert "Not determined" in section


class TestVulnerabilitiesSection:
    def test_vulnerabilities(self) -> None:
        landscape = _make_landscape()
        section = _vulnerabilities_section(landscape)
        assert "self_lobotomy_vulnerability" in section
        assert "Catastrophic" in section
        assert "Remediation" in section
        assert "Add immutable core functions" in section


class TestFragilityScoreSection:
    def test_fragility_section(self) -> None:
        report = _make_fragility_report()
        section = _fragility_score_section(report)
        assert "0.55" in section
        assert "C-" in section
        assert "Component Breakdown" in section
        assert "Recommendations" in section
        assert "Improve recovery mechanisms" in section


class TestDetailedResults:
    def test_detailed_results(self) -> None:
        results = _make_results()
        section = _detailed_results(results)
        assert "test_syntax" in section
        assert "test_logic" in section
        assert "test_self_ref" in section
        assert "PASS" in section
        assert "FAIL" in section
