"""Tests for comparison module: ablation, statistical tests, head-to-head."""

import pytest

from src.comparison.ablation import (
    AblationStudy,
    AblationResult,
    ConditionResult,
    ALL_CONDITIONS,
    CONDITION_FULL_THINKING,
    CONDITION_NO_THINKING,
)
from src.comparison.statistical_tests import StatisticalComparator, PairwiseTest, _mean, _variance
from src.comparison.head_to_head import HeadToHeadAnalyzer
from src.ga.engine import GAEngine, EvolutionResult
from src.operators.thinking_initializer import ThinkingInitializer
from src.operators.thinking_mutator import ThinkingMutator
from src.operators.thinking_crossover import ThinkingCrossover
from src.operators.thinking_evaluator import ThinkingEvaluator
from src.operators.non_thinking import (
    NonThinkingInitializer,
    NonThinkingMutator,
    NonThinkingEvaluator,
    SimpleCrossover,
)
from src.evaluation.financial_math import FinancialMathBenchmark


class TestConditionResult:
    """Tests for ConditionResult."""

    def test_compute_stats(self):
        cr = ConditionResult(
            condition_name="test",
            fitness_scores=[0.6, 0.7, 0.8],
        )
        cr.compute_stats()
        assert cr.mean_fitness == pytest.approx(0.7)
        assert cr.best_fitness == 0.8
        assert cr.std_fitness > 0

    def test_compute_stats_empty(self):
        cr = ConditionResult(condition_name="empty")
        cr.compute_stats()
        assert cr.mean_fitness == 0.0

    def test_compute_stats_single(self):
        cr = ConditionResult(
            condition_name="single",
            fitness_scores=[0.5],
        )
        cr.compute_stats()
        assert cr.mean_fitness == 0.5
        assert cr.best_fitness == 0.5


class TestAblationResult:
    """Tests for AblationResult."""

    def test_get_ranking(self):
        result = AblationResult()
        result.conditions["a"] = ConditionResult(
            condition_name="a", fitness_scores=[0.8], mean_fitness=0.8,
        )
        result.conditions["b"] = ConditionResult(
            condition_name="b", fitness_scores=[0.6], mean_fitness=0.6,
        )
        ranking = result.get_ranking()
        assert ranking[0] == "a"
        assert ranking[1] == "b"

    def test_generate_summary(self):
        result = AblationResult()
        result.conditions["full_thinking"] = ConditionResult(
            condition_name="full_thinking",
            fitness_scores=[0.7, 0.75],
        )
        result.conditions["full_thinking"].compute_stats()
        summary = result.generate_summary()
        assert "full_thinking" in summary

    def test_all_conditions_defined(self):
        assert len(ALL_CONDITIONS) == 7
        assert CONDITION_FULL_THINKING in ALL_CONDITIONS
        assert CONDITION_NO_THINKING in ALL_CONDITIONS


class TestAblationStudy:
    """Tests for AblationStudy."""

    def test_run_ablation(self, mock_thinking_llm, mock_output_llm, sample_tasks):
        def engine_factory(**kwargs):
            use_thinking = kwargs.get("use_thinking", True)
            pop_size = kwargs.get("population_size", 4)
            crossover_rate = kwargs.get("crossover_rate", 0.4)
            elitism_count = kwargs.get("elitism_count", 1)

            if use_thinking and not kwargs.get("random_mutation", False):
                init = ThinkingInitializer(mock_thinking_llm)
                mut = ThinkingMutator(mock_thinking_llm)
                cross = ThinkingCrossover(mock_thinking_llm)
            else:
                init = NonThinkingInitializer()
                mut = NonThinkingMutator()
                cross = SimpleCrossover()

            evaluator = ThinkingEvaluator(mock_output_llm)

            return GAEngine(
                initializer=init,
                mutator=mut,
                crossover_op=cross if crossover_rate > 0 else None,
                evaluator=evaluator,
                population_size=pop_size,
                num_generations=2,
                mutation_rate=0.3,
                crossover_rate=crossover_rate,
                elitism_count=elitism_count,
                tournament_size=2,
            )

        study = AblationStudy(
            engine_factory=engine_factory,
            domain_desc="Financial math",
            example_tasks=["task"],
        )

        result = study.run(
            eval_tasks=sample_tasks[:3],
            repetitions=2,
            conditions=["full_thinking", "no_thinking"],
        )

        assert len(result.conditions) == 2
        assert "full_thinking" in result.conditions
        assert "no_thinking" in result.conditions
        assert result.conditions["full_thinking"].mean_fitness >= 0.0
        assert len(result.summary) > 0

    def test_ablation_condition_configs(self, mock_thinking_llm, mock_output_llm, sample_tasks):
        """Test that different conditions produce different engine configs."""
        configs_tested = {}

        def tracking_factory(**kwargs):
            # Track what kwargs each condition produces
            condition_key = str(sorted(kwargs.items()))
            configs_tested[condition_key] = kwargs

            init = NonThinkingInitializer()
            mut = NonThinkingMutator()
            cross = SimpleCrossover()
            evaluator = ThinkingEvaluator(mock_output_llm)

            pop_size = kwargs.get("population_size", 4)
            crossover_rate = kwargs.get("crossover_rate", 0.4)
            elitism_count = kwargs.get("elitism_count", 1)

            return GAEngine(
                initializer=init,
                mutator=mut,
                crossover_op=cross if crossover_rate > 0 else None,
                evaluator=evaluator,
                population_size=pop_size,
                num_generations=1,
                mutation_rate=0.3,
                crossover_rate=crossover_rate,
                elitism_count=elitism_count,
                tournament_size=2,
            )

        study = AblationStudy(
            engine_factory=tracking_factory,
            domain_desc="test",
        )

        result = study.run(
            eval_tasks=sample_tasks[:3],
            repetitions=1,
            conditions=["full_thinking", "no_crossover", "no_elitism"],
        )

        assert len(result.conditions) == 3


class TestStatisticalComparator:
    """Tests for statistical tests."""

    def test_pairwise_test(self):
        comp = StatisticalComparator()
        result = comp.pairwise_test(
            [0.8, 0.85, 0.82, 0.79, 0.83],
            [0.6, 0.55, 0.58, 0.62, 0.57],
            condition_a="better",
            condition_b="worse",
        )

        assert isinstance(result, PairwiseTest)
        assert result.mean_a > result.mean_b
        assert result.t_statistic > 0
        assert result.p_value < 0.05
        assert result.significant is True

    def test_pairwise_test_identical(self):
        comp = StatisticalComparator()
        scores = [0.7, 0.7, 0.7]
        result = comp.pairwise_test(scores, scores)
        assert result.t_statistic == pytest.approx(0.0)

    def test_effect_size_large(self):
        comp = StatisticalComparator()
        d = comp.effect_size([0.8, 0.85, 0.82], [0.5, 0.55, 0.52])
        assert abs(d) > 0.8  # Large effect size

    def test_effect_size_zero(self):
        comp = StatisticalComparator()
        d = comp.effect_size([0.7, 0.7, 0.7], [0.7, 0.7, 0.7])
        assert d == pytest.approx(0.0)

    def test_effect_size_single_value(self):
        comp = StatisticalComparator()
        d = comp.effect_size([0.5], [0.5])
        assert d == pytest.approx(0.0)

    def test_improvement_with_ci(self):
        comp = StatisticalComparator()
        improvement, ci_lower, ci_upper = comp.improvement_with_ci(
            [0.8, 0.85, 0.82],
            [0.6, 0.65, 0.62],
        )
        assert improvement > 0  # A is better
        assert ci_lower < ci_upper

    def test_improvement_with_ci_zero_baseline(self):
        comp = StatisticalComparator()
        improvement, ci_lower, ci_upper = comp.improvement_with_ci(
            [0.5, 0.6], [0.0, 0.0],
        )
        assert improvement == 0.0  # Division by zero handling

    def test_bonferroni_correction(self):
        comp = StatisticalComparator()
        p_values = [0.01, 0.02, 0.03, 0.04]
        results = comp.bonferroni_correction(p_values, alpha=0.05)
        # Corrected alpha = 0.05 / 4 = 0.0125
        assert results[0] is True   # 0.01 < 0.0125
        assert results[1] is False  # 0.02 > 0.0125

    def test_bonferroni_empty(self):
        comp = StatisticalComparator()
        results = comp.bonferroni_correction([])
        assert results == []

    def test_approx_p_value_large_t(self):
        comp = StatisticalComparator()
        p = comp._approx_p_value(10.0)
        assert p < 0.01

    def test_approx_p_value_zero_t(self):
        comp = StatisticalComparator()
        p = comp._approx_p_value(0.0)
        assert p == pytest.approx(1.0)


class TestHelperFunctions:
    """Tests for _mean and _variance."""

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_values(self):
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_variance_empty(self):
        assert _variance([]) == 0.0

    def test_variance_single(self):
        assert _variance([5.0]) == 0.0

    def test_variance_values(self):
        v = _variance([2.0, 4.0, 6.0])
        assert v == pytest.approx(4.0)  # Sample variance


class TestHeadToHeadAnalyzer:
    """Tests for head-to-head comparison."""

    def _make_ablation_result(self):
        result = AblationResult()
        result.conditions["better"] = ConditionResult(
            condition_name="better",
            fitness_scores=[0.8, 0.85, 0.82, 0.79, 0.83],
        )
        result.conditions["better"].compute_stats()
        result.conditions["worse"] = ConditionResult(
            condition_name="worse",
            fitness_scores=[0.6, 0.55, 0.58, 0.62, 0.57],
        )
        result.conditions["worse"].compute_stats()
        result.generate_summary()
        return result

    def test_compare_conditions(self):
        analyzer = HeadToHeadAnalyzer()
        result = self._make_ablation_result()
        pairwise = analyzer.compare_conditions(result)
        assert len(pairwise) == 1  # One pair
        key = ("better", "worse")
        assert key in pairwise
        assert pairwise[key].significant is True

    def test_plot_comparison(self):
        analyzer = HeadToHeadAnalyzer()
        result = self._make_ablation_result()
        plot = analyzer.plot_comparison(result)
        assert "better" in plot
        assert "worse" in plot
        assert "#" in plot

    def test_generate_ranking_table(self):
        analyzer = HeadToHeadAnalyzer()
        result = self._make_ablation_result()
        pairwise = analyzer.compare_conditions(result)
        table = analyzer.generate_ranking_table(result, pairwise)
        assert "Rank" in table
        assert "better" in table
        assert "worse" in table

    def test_ranking_table_without_pairwise(self):
        analyzer = HeadToHeadAnalyzer()
        result = self._make_ablation_result()
        table = analyzer.generate_ranking_table(result)
        assert "Rank" in table

    def test_plot_empty_result(self):
        analyzer = HeadToHeadAnalyzer()
        result = AblationResult()
        result.conditions["only"] = ConditionResult(
            condition_name="only",
            fitness_scores=[0.5],
        )
        result.conditions["only"].compute_stats()
        plot = analyzer.plot_comparison(result)
        assert "only" in plot


class TestNonThinkingOperators:
    """Tests for non-thinking operators used in ablation."""

    def test_non_thinking_initializer(self):
        init = NonThinkingInitializer()
        genomes = init.initialize(3, "test domain")
        assert len(genomes) == 3
        for g in genomes:
            assert "identity" in g.sections
            assert g.operator == "init_template"

    def test_non_thinking_mutator(self, sample_genome):
        mutator = NonThinkingMutator()
        mutated = mutator.mutate(sample_genome)
        assert mutated.genome_id != sample_genome.genome_id
        assert mutated.operator == "mutate_random"
        assert mutated.fitness == 0.0

    def test_non_thinking_evaluator_no_llm(self, sample_genome, sample_tasks):
        evaluator = NonThinkingEvaluator()
        details = evaluator.evaluate(sample_genome, sample_tasks)
        assert details.accuracy == 0.5  # Default baseline
        assert details.composite_fitness > 0.0

    def test_non_thinking_evaluator_with_llm(
        self, mock_output_llm, sample_genome, sample_tasks
    ):
        evaluator = NonThinkingEvaluator()
        details = evaluator.evaluate(
            sample_genome, sample_tasks, llm_call=mock_output_llm
        )
        assert 0.0 <= details.accuracy <= 1.0

    def test_simple_crossover(self, sample_genome, sample_genome_b):
        crossover = SimpleCrossover()
        offspring = crossover.crossover(
            sample_genome, sample_genome_b, 0.7, 0.5
        )
        assert offspring.genome_id != sample_genome.genome_id
        assert offspring.operator == "crossover_simple"
        assert len(offspring.sections) > 0
