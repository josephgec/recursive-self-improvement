"""Tests for LaTeX table and figure generation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.suites.base import AblationSuiteResult, ConditionRun
from src.publication.latex_tables import LaTeXTableGenerator
from src.publication.figures import PublicationFigureGenerator, FigureData, COLORBLIND_PALETTE
from src.publication.significance_stars import add_stars
from src.analysis.statistical_tests import PublicationStatistics, PairwiseResult


@pytest.fixture
def table_gen():
    return LaTeXTableGenerator()


@pytest.fixture
def figure_gen():
    return PublicationFigureGenerator()


@pytest.fixture
def stats():
    return PublicationStatistics()


class TestSignificanceStars:
    """Test significance star assignment."""

    def test_three_stars(self):
        assert add_stars(0.0001) == "***"
        assert add_stars(0.0005) == "***"

    def test_two_stars(self):
        assert add_stars(0.005) == "**"
        assert add_stars(0.001) == "**"

    def test_one_star(self):
        assert add_stars(0.01) == "*"
        assert add_stars(0.04) == "*"

    def test_no_stars(self):
        assert add_stars(0.05) == ""
        assert add_stars(0.10) == ""
        assert add_stars(0.50) == ""
        assert add_stars(1.0) == ""

    def test_boundary_values(self):
        assert add_stars(0.001) == "**"   # exactly 0.001 is not < 0.001
        assert add_stars(0.01) == "*"     # exactly 0.01 is not < 0.01
        assert add_stars(0.05) == ""      # exactly 0.05 is not < 0.05


class TestMainResultsTable:
    """Test main results table generation."""

    def test_valid_latex(self, table_gen, sample_result):
        table = table_gen.main_results_table(sample_result)
        assert "\\begin{table}" in table
        assert "\\end{table}" in table
        assert "\\toprule" in table
        assert "\\midrule" in table
        assert "\\bottomrule" in table

    def test_contains_all_conditions(self, table_gen, sample_result):
        table = table_gen.main_results_table(sample_result)
        assert "full" in table
        assert "ablated" in table

    def test_bold_best(self, table_gen, sample_result):
        table = table_gen.main_results_table(sample_result)
        # Full is best; should be bold
        assert "\\textbf{full}" in table

    def test_custom_caption(self, table_gen, sample_result):
        table = table_gen.main_results_table(
            sample_result, caption="My Custom Caption"
        )
        assert "My Custom Caption" in table

    def test_custom_label(self, table_gen, sample_result):
        table = table_gen.main_results_table(
            sample_result, label="tab:custom"
        )
        assert "tab:custom" in table

    def test_empty_result(self, table_gen):
        result = AblationSuiteResult(suite_name="Empty")
        table = table_gen.main_results_table(result)
        assert table == ""

    def test_has_accuracy_values(self, table_gen, sample_result):
        table = table_gen.main_results_table(sample_result)
        # Should contain mean accuracy formatted to 3 decimal places
        assert "0.850" in table or "0.85" in table


class TestPairwiseComparisonTable:
    """Test pairwise comparison table generation."""

    def test_valid_latex(self, table_gen, stats, sample_result):
        scores_a = sample_result.get_scores("full")
        scores_b = sample_result.get_scores("ablated")
        pw = stats.pairwise_comparison(scores_a, scores_b, "full", "ablated")
        table = table_gen.pairwise_comparison_table([pw])

        assert "\\begin{table}" in table
        assert "\\end{table}" in table
        assert "\\toprule" in table
        assert "\\bottomrule" in table

    def test_contains_comparison_names(self, table_gen, stats, sample_result):
        scores_a = sample_result.get_scores("full")
        scores_b = sample_result.get_scores("ablated")
        pw = stats.pairwise_comparison(scores_a, scores_b, "full", "ablated")
        table = table_gen.pairwise_comparison_table([pw])

        assert "full" in table
        assert "ablated" in table

    def test_contains_stars(self, table_gen, stats, sample_result):
        scores_a = sample_result.get_scores("full")
        scores_b = sample_result.get_scores("ablated")
        pw = stats.pairwise_comparison(scores_a, scores_b, "full", "ablated")
        table = table_gen.pairwise_comparison_table([pw])
        # The difference is large enough to be significant
        # so there should be stars
        assert "*" in table

    def test_empty_comparisons(self, table_gen):
        table = table_gen.pairwise_comparison_table([])
        assert table == ""

    def test_contains_ci(self, table_gen, stats, sample_result):
        scores_a = sample_result.get_scores("full")
        scores_b = sample_result.get_scores("ablated")
        pw = stats.pairwise_comparison(scores_a, scores_b, "full", "ablated")
        table = table_gen.pairwise_comparison_table([pw])
        # Should contain CI brackets
        assert "[" in table and "]" in table


class TestConditionDetailTable:
    """Test per-condition detail tables."""

    def test_valid_latex(self, table_gen, sample_result):
        table = table_gen.condition_detail_table(sample_result, "full")
        assert "\\begin{table}" in table
        assert "\\end{table}" in table

    def test_shows_per_run_scores(self, table_gen, sample_result):
        table = table_gen.condition_detail_table(sample_result, "full")
        # Should have run numbers
        assert "1 &" in table
        assert "2 &" in table

    def test_nonexistent_condition(self, table_gen, sample_result):
        table = table_gen.condition_detail_table(sample_result, "nonexistent")
        assert table == ""


class TestFigureGeneration:
    """Test figure generation (data mode, no matplotlib required)."""

    def test_bar_chart_returns_figure_data(self, figure_gen, sample_result):
        fig = figure_gen.ablation_bar_chart(sample_result)
        assert isinstance(fig, FigureData)
        assert fig.fig_type == "bar_chart"
        assert "conditions" in fig.data
        assert "means" in fig.data

    def test_improvement_curve(self, figure_gen, sample_result):
        results = {"Suite A": sample_result}
        fig = figure_gen.improvement_curve_comparison(results)
        assert isinstance(fig, FigureData)
        assert fig.fig_type == "improvement_curve"

    def test_waterfall_chart(self, figure_gen, sample_result):
        fig = figure_gen.contribution_waterfall(sample_result)
        assert isinstance(fig, FigureData)
        assert fig.fig_type == "waterfall"
        assert "contributions" in fig.data

    def test_forest_plot(self, figure_gen, stats, sample_result):
        scores_a = sample_result.get_scores("full")
        scores_b = sample_result.get_scores("ablated")
        pw = stats.pairwise_comparison(scores_a, scores_b, "full", "ablated")
        fig = figure_gen.pairwise_forest_plot([pw])
        assert isinstance(fig, FigureData)
        assert fig.fig_type == "forest_plot"

    def test_colorblind_palette(self):
        assert len(COLORBLIND_PALETTE) >= 8

    def test_get_colors(self, figure_gen):
        colors = figure_gen._get_colors(5)
        assert len(colors) == 5
        # Should be from the colorblind palette
        assert all(c in COLORBLIND_PALETTE for c in colors)

    def test_bar_chart_data_has_correct_length(self, figure_gen, sample_result):
        fig = figure_gen.ablation_bar_chart(sample_result)
        n = len(sample_result.get_all_condition_names())
        assert len(fig.data["conditions"]) == n
        assert len(fig.data["means"]) == n
        assert len(fig.data["stds"]) == n


class TestFigureGenerationWithFiles:
    """Test figure generation that actually saves files (matplotlib paths)."""

    def test_bar_chart_saves_file(self, figure_gen, sample_result, tmp_path):
        output = str(tmp_path / "bar.png")
        fig = figure_gen.ablation_bar_chart(sample_result, output_path=output)
        if figure_gen._has_matplotlib:
            assert fig.saved_path == output
            assert os.path.exists(output)

    def test_improvement_curve_saves_file(self, figure_gen, sample_result, tmp_path):
        output = str(tmp_path / "curve.png")
        results = {"Suite A": sample_result}
        fig = figure_gen.improvement_curve_comparison(results, output_path=output)
        if figure_gen._has_matplotlib:
            assert fig.saved_path == output

    def test_waterfall_saves_file(self, figure_gen, sample_result, tmp_path):
        output = str(tmp_path / "waterfall.png")
        fig = figure_gen.contribution_waterfall(sample_result, output_path=output)
        if figure_gen._has_matplotlib:
            assert fig.saved_path == output

    def test_forest_plot_saves_file(self, figure_gen, stats, sample_result, tmp_path):
        output = str(tmp_path / "forest.png")
        scores_a = sample_result.get_scores("full")
        scores_b = sample_result.get_scores("ablated")
        pw = stats.pairwise_comparison(scores_a, scores_b, "full", "ablated")
        fig = figure_gen.pairwise_forest_plot([pw], output_path=output)
        if figure_gen._has_matplotlib:
            assert fig.saved_path == output

    def test_bar_chart_custom_title(self, figure_gen, sample_result, tmp_path):
        output = str(tmp_path / "bar_titled.png")
        fig = figure_gen.ablation_bar_chart(
            sample_result, title="Custom Title", output_path=output
        )
        assert fig.title == "Custom Title"

    def test_figure_data_repr(self):
        from src.publication.figures import FigureData
        fd = FigureData(fig_type="test", data={}, title="Test Figure")
        r = repr(fd)
        assert "test" in r
        assert "Test Figure" in r

    def test_no_colorblind(self, sample_result, tmp_path):
        gen = PublicationFigureGenerator(colorblind_safe=False)
        colors = gen._get_colors(3)
        assert len(colors) == 3
        # Without colorblind, uses Cn notation
        assert colors[0] == "C0"

    def test_apply_style(self, figure_gen, tmp_path):
        """Test _apply_style runs without error."""
        if figure_gen._has_matplotlib:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            figure_gen._apply_style(ax)
            plt.close(fig)
