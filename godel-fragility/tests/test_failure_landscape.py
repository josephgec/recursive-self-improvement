"""Tests for the failure landscape analyzer."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from src.analysis.failure_landscape import (
    FailureLandscape,
    FailureLandscapeAnalyzer,
    Vulnerability,
)
from src.harness.stress_runner import ScenarioResult, StressTestResults
from src.measurement.failure_classifier import FailureMode


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


def _make_scenario_result(
    name: str,
    category: str,
    severity: int,
    success: bool,
    failure_mode: FailureMode,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=name,
        category=category,
        severity=severity,
        success=success,
        failure_mode=failure_mode,
        accuracies=[0.8, 0.5, 0.3],
        complexities=[30, 50, 70],
        iterations_run=3,
        duration_seconds=1.0,
    )


def _make_stress_results() -> StressTestResults:
    results = [
        _make_scenario_result("s1", "fault_injection", 3, True, FailureMode.VALIDATION_CAUGHT),
        _make_scenario_result("s2", "fault_injection", 4, False, FailureMode.SILENT_DEGRADATION),
        _make_scenario_result("s3", "self_reference", 5, False, FailureMode.SELF_LOBOTOMY),
        _make_scenario_result("s4", "self_reference", 4, False, FailureMode.RUNAWAY_MODIFICATION),
        _make_scenario_result("s5", "complexity", 3, False, FailureMode.COMPLEXITY_EXPLOSION),
    ]
    return StressTestResults(
        results=results,
        total_scenarios=5,
        passed=1,
        failed=4,
        timed_out=0,
        duration_seconds=5.0,
    )


@pytest.fixture
def analyzer() -> FailureLandscapeAnalyzer:
    return FailureLandscapeAnalyzer()


@pytest.fixture
def stress_results() -> StressTestResults:
    return _make_stress_results()


# ------------------------------------------------------------------ #
# compute_landscape
# ------------------------------------------------------------------ #


class TestComputeLandscape:
    def test_failure_mode_counts(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults
    ) -> None:
        landscape = analyzer.compute_landscape(stress_results)
        assert landscape.total_scenarios == 5
        assert landscape.total_failures == 4
        assert landscape.failure_mode_counts["silent_degradation"] == 1
        assert landscape.failure_mode_counts["self_lobotomy"] == 1
        assert landscape.failure_mode_counts["runaway_modification"] == 1
        assert landscape.failure_mode_counts["complexity_explosion"] == 1

    def test_category_failure_rates(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults
    ) -> None:
        landscape = analyzer.compute_landscape(stress_results)
        assert landscape.category_failure_rates["fault_injection"] == pytest.approx(0.5)
        assert landscape.category_failure_rates["self_reference"] == pytest.approx(1.0)
        assert landscape.category_failure_rates["complexity"] == pytest.approx(1.0)

    def test_severity_distribution(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults
    ) -> None:
        landscape = analyzer.compute_landscape(stress_results)
        assert landscape.severity_distribution[3] == 1  # complexity_explosion
        assert landscape.severity_distribution[4] == 2  # silent_deg + runaway
        assert landscape.severity_distribution[5] == 1  # self_lobotomy

    def test_heatmap_data(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults
    ) -> None:
        landscape = analyzer.compute_landscape(stress_results)
        assert landscape.heatmap_data is not None
        assert "self_reference" in landscape.heatmap_data
        assert landscape.heatmap_data["self_reference"]["self_lobotomy"] == 1

    def test_vulnerabilities_identified(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults
    ) -> None:
        landscape = analyzer.compute_landscape(stress_results)
        # Should find vulnerabilities for severity >= 3 modes
        assert len(landscape.vulnerabilities) > 0
        vuln_names = [v.name for v in landscape.vulnerabilities]
        assert any("self_lobotomy" in n for n in vuln_names)

    def test_all_passing(self, analyzer: FailureLandscapeAnalyzer) -> None:
        results = StressTestResults(
            results=[
                _make_scenario_result("s1", "test", 1, True, FailureMode.STAGNATION),
            ],
            total_scenarios=1,
            passed=1,
            failed=0,
        )
        landscape = analyzer.compute_landscape(results)
        assert landscape.total_failures == 0
        assert landscape.failure_mode_counts == {}


# ------------------------------------------------------------------ #
# identify_critical_vulnerabilities
# ------------------------------------------------------------------ #


class TestIdentifyCriticalVulnerabilities:
    def test_critical_vulns(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults
    ) -> None:
        vulns = analyzer.identify_critical_vulnerabilities(stress_results)
        assert isinstance(vulns, list)
        # Should be sorted by severity descending
        severities = [v.severity for v in vulns]
        assert severities == sorted(severities, reverse=True)

    def test_remediation_present(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults
    ) -> None:
        vulns = analyzer.identify_critical_vulnerabilities(stress_results)
        for v in vulns:
            assert v.remediation  # non-empty

    def test_no_failures_no_vulns(self, analyzer: FailureLandscapeAnalyzer) -> None:
        results = StressTestResults(
            results=[
                _make_scenario_result("s1", "test", 1, True, FailureMode.STAGNATION),
            ],
            total_scenarios=1,
            passed=1,
            failed=0,
        )
        vulns = analyzer.identify_critical_vulnerabilities(results)
        assert vulns == []


# ------------------------------------------------------------------ #
# plot_failure_heatmap
# ------------------------------------------------------------------ #


class TestPlotFailureHeatmap:
    def test_plot_heatmap(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults
    ) -> None:
        import matplotlib.pyplot as plt

        landscape = analyzer.compute_landscape(stress_results)
        fig = analyzer.plot_failure_heatmap(landscape)
        assert fig is not None
        plt.close(fig)

    def test_plot_empty_heatmap(self, analyzer: FailureLandscapeAnalyzer) -> None:
        landscape = FailureLandscape()
        fig = analyzer.plot_failure_heatmap(landscape)
        assert fig is None

    def test_plot_saves_to_file(
        self, analyzer: FailureLandscapeAnalyzer, stress_results: StressTestResults, tmp_path
    ) -> None:
        import matplotlib.pyplot as plt
        import os

        landscape = analyzer.compute_landscape(stress_results)
        output_path = str(tmp_path / "heatmap.png")
        fig = analyzer.plot_failure_heatmap(landscape, output_path=output_path)
        assert fig is not None
        assert os.path.exists(output_path)
        plt.close(fig)


# ------------------------------------------------------------------ #
# _suggest_remediation
# ------------------------------------------------------------------ #


class TestSuggestRemediation:
    def test_known_modes(self, analyzer: FailureLandscapeAnalyzer) -> None:
        known_modes = [
            "self_lobotomy",
            "state_corruption",
            "silent_degradation",
            "runaway_modification",
            "rollback_failure",
            "complexity_explosion",
            "infinite_loop",
            "oscillation",
            "stagnation",
        ]
        for mode in known_modes:
            suggestion = analyzer._suggest_remediation(mode)
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0

    def test_unknown_mode(self, analyzer: FailureLandscapeAnalyzer) -> None:
        suggestion = analyzer._suggest_remediation("unknown_mode")
        assert "Review agent architecture" in suggestion


# ------------------------------------------------------------------ #
# Vulnerability dataclass
# ------------------------------------------------------------------ #


class TestVulnerability:
    def test_fields(self) -> None:
        v = Vulnerability(
            name="test_vuln",
            category="test",
            severity=4,
            description="A test vulnerability",
            failure_mode=FailureMode.SILENT_DEGRADATION,
            affected_scenarios=["s1", "s2"],
            remediation="Fix it.",
        )
        assert v.name == "test_vuln"
        assert v.severity == 4
        assert len(v.affected_scenarios) == 2
