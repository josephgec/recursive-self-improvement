"""Tests for analysis modules: complexity landscape, library growth, escape, report."""

import os

import pytest

from src.bdm.scorer import BDMScorer
from src.bdm.calibration import BDMCalibrator, CalibrationReport
from src.analysis.complexity_landscape import ComplexityLandscapeAnalyzer, DomainComplexity
from src.analysis.library_growth import LibraryGrowthAnalyzer
from src.analysis.escape_analysis import CollapseEscapeAnalyzer, CollapseEscapeResult
from src.analysis.report import generate_report
from src.library.evolution import LibraryMetrics


class TestComplexityLandscape:
    """Tests for complexity landscape analysis."""

    def test_map_bdm_across_domains(self, scorer):
        """Should compute complexity stats for each domain."""
        analyzer = ComplexityLandscapeAnalyzer(scorer=scorer)
        domain_data = {
            "constant": ["0000", "1111", "0000"],
            "periodic": ["0101", "0011", "1010"],
            "random": ["0110", "1001", "0111"],
        }
        results = analyzer.map_bdm_across_domains(domain_data)
        assert len(results) == 3
        for dc in results:
            assert isinstance(dc, DomainComplexity)
            assert dc.samples > 0
            assert dc.avg_bdm > 0

    def test_constant_lower_than_random(self, scorer):
        """Constant domain should have lower avg BDM than random."""
        analyzer = ComplexityLandscapeAnalyzer(scorer=scorer)
        domain_data = {
            "constant": ["0000", "1111", "0000", "1111"],
            "random": ["0110", "1001", "0111", "1010"],
        }
        results = analyzer.map_bdm_across_domains(domain_data)
        constant_dc = next(r for r in results if r.domain == "constant")
        random_dc = next(r for r in results if r.domain == "random")
        assert constant_dc.avg_bdm <= random_dc.avg_bdm

    def test_empty_domain_skipped(self, scorer):
        """Empty domains should be skipped."""
        analyzer = ComplexityLandscapeAnalyzer(scorer=scorer)
        results = analyzer.map_bdm_across_domains({"empty": [], "notempty": ["0101"]})
        assert len(results) == 1

    def test_plot_landscape(self, scorer, tmp_path):
        """Plotting should not crash."""
        analyzer = ComplexityLandscapeAnalyzer(scorer=scorer)
        domain_data = {
            "constant": ["0000", "1111"],
            "random": ["0110", "1001"],
        }
        results = analyzer.map_bdm_across_domains(domain_data)
        output = str(tmp_path / "landscape.png")
        analyzer.plot_landscape(results, output_path=output)
        # File may or may not exist depending on matplotlib


class TestLibraryGrowth:
    """Tests for library growth analysis."""

    def test_add_snapshot(self):
        """Adding snapshots should track them."""
        analyzer = LibraryGrowthAnalyzer()
        metrics = LibraryMetrics(
            total_rules=10, unique_domains=3, avg_accuracy=0.8, quality_score=0.6
        )
        analyzer.add_snapshot(metrics)
        assert len(analyzer._snapshots) == 1

    def test_domain_coverage_over_time(self):
        """Should return domain counts at each snapshot."""
        analyzer = LibraryGrowthAnalyzer()
        for i in range(5):
            metrics = LibraryMetrics(total_rules=i * 5, unique_domains=i + 1)
            analyzer.add_snapshot(metrics)

        coverage = analyzer.domain_coverage_over_time()
        assert len(coverage) == 5
        assert coverage == [1, 2, 3, 4, 5]

    def test_plot_library_growth(self, tmp_path):
        """Plotting growth should not crash."""
        analyzer = LibraryGrowthAnalyzer()
        for i in range(5):
            metrics = LibraryMetrics(
                total_rules=i * 5,
                unique_domains=i + 1,
                avg_accuracy=0.5 + 0.1 * i,
                quality_score=0.3 + 0.1 * i,
            )
            analyzer.add_snapshot(metrics)

        output = str(tmp_path / "growth.png")
        analyzer.plot_library_growth(output_path=output)

    def test_plot_empty_snapshots(self):
        """Plotting with no data should return None."""
        analyzer = LibraryGrowthAnalyzer()
        result = analyzer.plot_library_growth()
        assert result is None

    def test_add_pareto_snapshot(self):
        """Adding Pareto snapshots should track them."""
        analyzer = LibraryGrowthAnalyzer()
        front = [{"accuracy": 0.9, "complexity": 10.0}, {"accuracy": 0.7, "complexity": 5.0}]
        analyzer.add_pareto_snapshot(front)
        assert len(analyzer._pareto_history) == 1

    def test_plot_pareto_evolution(self, tmp_path):
        """Plotting Pareto evolution should not crash."""
        analyzer = LibraryGrowthAnalyzer()
        for i in range(3):
            front = [
                {"accuracy": 0.5 + 0.1 * i, "complexity": 10.0 - i},
                {"accuracy": 0.8 + 0.05 * i, "complexity": 15.0 - i},
            ]
            analyzer.add_pareto_snapshot(front)

        output = str(tmp_path / "pareto_evolution.png")
        analyzer.plot_pareto_front_evolution(output_path=output)

    def test_plot_pareto_evolution_empty(self):
        """Plotting empty Pareto history should return None."""
        analyzer = LibraryGrowthAnalyzer()
        result = analyzer.plot_pareto_front_evolution()
        assert result is None


class TestCollapseEscapeAnalysis:
    """Tests for collapse/escape analysis."""

    def test_analyze_escaping(self):
        """Improving metrics should show escaping."""
        analyzer = CollapseEscapeAnalyzer(window_size=5)
        history = [
            LibraryMetrics(
                total_rules=i * 10,
                unique_domains=i + 1,
                avg_accuracy=0.5 + 0.05 * i,
                quality_score=0.3 + 0.1 * i,
                coverage=0.3 + 0.1 * i,
            )
            for i in range(5)
        ]
        results = analyzer.analyze(history)
        assert len(results) > 0

        # At least some metrics should show escaping
        escaping = [r for r in results if r.is_escaping]
        assert len(escaping) > 0

    def test_analyze_collapsing(self):
        """Declining metrics should show collapsing."""
        analyzer = CollapseEscapeAnalyzer(window_size=5)
        history = [
            LibraryMetrics(
                total_rules=100 - i * 20,
                unique_domains=5 - i,
                avg_accuracy=0.9 - 0.1 * i,
                quality_score=0.8 - 0.15 * i,
                coverage=0.9 - 0.1 * i,
            )
            for i in range(5)
        ]
        results = analyzer.analyze(history)
        collapsing = [r for r in results if r.is_collapsing]
        assert len(collapsing) > 0

    def test_analyze_stable(self):
        """Stable metrics should show neither collapsing nor escaping."""
        analyzer = CollapseEscapeAnalyzer(window_size=5)
        history = [
            LibraryMetrics(
                total_rules=50,
                unique_domains=3,
                avg_accuracy=0.8,
                quality_score=0.6,
                coverage=0.7,
            )
            for _ in range(5)
        ]
        results = analyzer.analyze(history)
        for r in results:
            assert not r.is_collapsing
            assert not r.is_escaping
            assert r.status == "stable"

    def test_analyze_too_few_points(self):
        """Less than 2 data points should return empty."""
        analyzer = CollapseEscapeAnalyzer()
        results = analyzer.analyze([LibraryMetrics()])
        assert results == []

    def test_result_status_property(self):
        """Status property should return correct strings."""
        r = CollapseEscapeResult(metric_name="test", is_escaping=True)
        assert r.status == "escaping"

        r = CollapseEscapeResult(metric_name="test", is_collapsing=True)
        assert r.status == "collapsing"

        r = CollapseEscapeResult(metric_name="test")
        assert r.status == "stable"

    def test_plot_escape_vs_collapse(self, tmp_path):
        """Plotting should not crash."""
        analyzer = CollapseEscapeAnalyzer()
        results = [
            CollapseEscapeResult(
                metric_name="accuracy",
                values=[0.5, 0.6, 0.7, 0.8],
                is_escaping=True,
                trend=0.1,
            ),
            CollapseEscapeResult(
                metric_name="size",
                values=[10, 12, 15, 18],
                is_escaping=True,
                trend=2.7,
            ),
        ]
        output = str(tmp_path / "escape.png")
        analyzer.plot_escape_vs_collapse(results, output_path=output)

    def test_plot_empty_results(self):
        """Plotting empty results should return None."""
        analyzer = CollapseEscapeAnalyzer()
        result = analyzer.plot_escape_vs_collapse([])
        assert result is None


class TestReport:
    """Tests for report generation."""

    def test_generate_basic_report(self):
        """Should generate a valid markdown report."""
        report = generate_report()
        assert "# CTM Integration Report" in report
        assert "## 1. BDM Calibration" in report
        assert "## 5. Summary" in report

    def test_report_with_calibration(self):
        """Report should include calibration data."""
        scorer = BDMScorer(block_size=4)
        calibrator = BDMCalibrator(scorer)
        cal_report = calibrator.run_calibration(length=16)

        report = generate_report(calibration_report=cal_report)
        assert "Ordering correct" in report
        assert "constant_zeros" in report

    def test_report_with_metrics(self):
        """Report should include library metrics."""
        metrics = LibraryMetrics(
            total_rules=42,
            unique_domains=5,
            avg_accuracy=0.85,
            avg_bdm_score=15.0,
            avg_mdl_score=20.0,
            coverage=0.8,
            quality_score=0.72,
        )
        report = generate_report(library_metrics=metrics)
        assert "42" in report
        assert "85" in report  # 85% accuracy

    def test_report_with_synthesis_trajectory(self):
        """Report should include synthesis trajectory."""
        trajectory = [
            {"iteration": 0, "candidates": 5, "best_accuracy": 0.6, "pareto_size": 2},
            {"iteration": 1, "candidates": 8, "best_accuracy": 0.9, "pareto_size": 3},
        ]
        report = generate_report(synthesis_trajectory=trajectory)
        assert "Synthesis Trajectory" in report

    def test_report_with_escape_results(self):
        """Report should include escape analysis."""
        escape_results = [
            CollapseEscapeResult(
                metric_name="accuracy",
                values=[0.5, 0.6, 0.7],
                is_escaping=True,
                trend=0.1,
            ),
        ]
        report = generate_report(escape_results=escape_results)
        assert "Collapse/Escape" in report
        assert "escaping" in report

    def test_report_save_to_file(self, tmp_path):
        """Report should save to file."""
        path = str(tmp_path / "test_report.md")
        report = generate_report(output_path=path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert content == report

    def test_report_collapsing_warning(self):
        """Report should warn about collapsing metrics."""
        escape_results = [
            CollapseEscapeResult(
                metric_name="quality",
                values=[0.8, 0.6, 0.4],
                is_collapsing=True,
                trend=-0.2,
            ),
        ]
        report = generate_report(escape_results=escape_results)
        assert "Warning" in report
