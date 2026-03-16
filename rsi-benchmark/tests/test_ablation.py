"""Test ablation: 7 conditions, contribution scoring, synergy, ranking, waterfall."""

import pytest

from tests.conftest import MockAgent
from src.ablation.ablation_study import ParadigmAblationStudy, AblationResult, AblationRun
from src.ablation.conditions import (
    AblationCondition,
    build_all_conditions,
    configure_pipeline_for_condition,
    ALL_COMPONENTS,
)
from src.ablation.contribution import ContributionAnalyzer, ParadigmContribution
from src.ablation.interaction_effects import InteractionAnalyzer, InteractionEffect


class TestAblationConditions:
    """Test ablation condition definitions."""

    def test_build_all_conditions(self):
        conditions = build_all_conditions()
        assert len(conditions) == 7
        names = [c.name for c in conditions]
        assert "full_pipeline" in names
        assert "no_soar" in names
        assert "no_ctm" in names
        assert "no_godel" in names
        assert "no_rlm" in names
        assert "soar_only" in names
        assert "naive_self_train" in names

    def test_full_pipeline_has_all_components(self):
        conditions = build_all_conditions()
        full = [c for c in conditions if c.name == "full_pipeline"][0]
        assert set(full.enabled_components) == set(ALL_COMPONENTS)
        assert full.disabled_components == []

    def test_no_soar_condition(self):
        conditions = build_all_conditions()
        no_soar = [c for c in conditions if c.name == "no_soar"][0]
        assert "soar" not in no_soar.enabled_components
        assert "soar" in no_soar.disabled_components

    def test_naive_has_no_components(self):
        conditions = build_all_conditions()
        naive = [c for c in conditions if c.name == "naive_self_train"][0]
        assert naive.enabled_components == []
        assert set(naive.disabled_components) == set(ALL_COMPONENTS)
        assert naive.expected_behavior == "degrading"

    def test_configure_pipeline(self):
        conditions = build_all_conditions()
        full = [c for c in conditions if c.name == "full_pipeline"][0]
        config = configure_pipeline_for_condition(full)
        assert config["use_soar"] is True
        assert config["use_ctm"] is True
        assert config["use_godel"] is True
        assert config["use_rlm"] is True

    def test_configure_pipeline_no_soar(self):
        conditions = build_all_conditions()
        no_soar = [c for c in conditions if c.name == "no_soar"][0]
        config = configure_pipeline_for_condition(no_soar)
        assert config["use_soar"] is False
        assert config["use_ctm"] is True


class TestParadigmAblationStudy:
    """Test running the ablation study."""

    def test_run_ablation(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=5,
        )
        result = study.run(sample_benchmarks)
        assert isinstance(result, AblationResult)
        assert len(result.conditions) == 7
        assert len(result.benchmarks) == 2

    def test_ablation_result_structure(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=5,
        )
        result = study.run(sample_benchmarks)

        # Check runs structure
        for cond in result.conditions:
            assert cond in result.runs
            for bm in result.benchmarks:
                assert bm in result.runs[cond]
                run = result.runs[cond][bm]
                assert isinstance(run, AblationRun)
                assert len(run.iterations) == 5
                assert len(run.accuracies) == 5

    def test_full_pipeline_best_improvement(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=10,
        )
        result = study.run(sample_benchmarks)

        # Full pipeline should have best improvement
        assert result.summary["full_pipeline"] >= result.summary["no_soar"]
        assert result.summary["full_pipeline"] >= result.summary["no_godel"]

    def test_naive_self_train_declines(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=10,
        )
        result = study.run(sample_benchmarks)

        # Naive self-train should decline
        assert result.summary["naive_self_train"] < 0

    def test_summary_values(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=5,
        )
        result = study.run(sample_benchmarks)
        assert len(result.summary) == 7

    def test_custom_conditions(self, sample_benchmarks):
        custom = [
            AblationCondition(
                name="custom_test",
                description="Custom condition",
                enabled_components=["soar"],
                disabled_components=["ctm", "godel", "rlm"],
                expected_behavior="slow_improving",
            )
        ]
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent("soar_only"),
            num_iterations=3,
            conditions=custom,
        )
        result = study.run(sample_benchmarks)
        assert len(result.conditions) == 1
        assert "custom_test" in result.runs


class TestContributionAnalyzer:
    """Test contribution analysis."""

    def _get_result(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=10,
        )
        return study.run(sample_benchmarks)

    def test_compute_contributions(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = ContributionAnalyzer()
        contributions = analyzer.compute_contributions(result)

        assert "soar" in contributions
        assert "ctm" in contributions
        assert "godel" in contributions
        assert "rlm" in contributions

        for c in contributions.values():
            assert isinstance(c, ParadigmContribution)
            # Marginal should be positive (removing hurts)
            assert c.marginal_contribution >= 0

    def test_relative_contributions_sum(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = ContributionAnalyzer()
        contributions = analyzer.compute_contributions(result)
        # Relative contributions can sum to more than 1 (synergy) or less
        total_relative = sum(c.relative_contribution for c in contributions.values())
        assert total_relative > 0

    def test_compute_synergy(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = ContributionAnalyzer()
        synergy = analyzer.compute_synergy(result)
        # Synergy is a float (can be positive or negative)
        assert isinstance(synergy, float)

    def test_rank_paradigms(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = ContributionAnalyzer()
        ranked = analyzer.rank_paradigms(result)

        assert len(ranked) == 4
        assert ranked[0].rank == 1
        assert ranked[-1].rank == 4
        # Contributions should be in descending order
        for i in range(len(ranked) - 1):
            assert ranked[i].marginal_contribution >= ranked[i + 1].marginal_contribution

    def test_plot_contribution_waterfall(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = ContributionAnalyzer()
        plot = analyzer.plot_contribution_waterfall(result)

        assert "labels" in plot
        assert "values" in plot
        assert "cumulative" in plot
        assert "full_improvement" in plot
        assert plot["labels"][0] == "naive_baseline"
        assert plot["labels"][-1] == "full_pipeline"

    def test_plot_ablation_curves(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = ContributionAnalyzer()
        plot = analyzer.plot_ablation_curves(result)

        assert "full_pipeline" in plot
        assert "naive_self_train" in plot
        assert len(plot["full_pipeline"]) == 10


class TestInteractionAnalyzer:
    """Test interaction effect analysis."""

    def _get_result(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=10,
        )
        return study.run(sample_benchmarks)

    def test_detect_synergy(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = InteractionAnalyzer()
        effects = analyzer.detect_synergy(result)
        assert isinstance(effects, list)
        for e in effects:
            assert isinstance(e, InteractionEffect)
            assert e.effect_type == "synergy"

    def test_detect_redundancy(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = InteractionAnalyzer()
        effects = analyzer.detect_redundancy(result)
        assert isinstance(effects, list)
        for e in effects:
            assert isinstance(e, InteractionEffect)
            assert e.effect_type == "redundancy"

    def test_analyze_all(self, sample_benchmarks):
        result = self._get_result(sample_benchmarks)
        analyzer = InteractionAnalyzer()
        analysis = analyzer.analyze_all(result)
        assert "synergy" in analysis
        assert "redundancy" in analysis
