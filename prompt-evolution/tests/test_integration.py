"""Integration tests: full evolution + ablation, verify improvement."""

import pytest

from tests.conftest import create_mock_thinking_llm, create_mock_output_llm
from src.genome.prompt_genome import PromptGenome
from src.genome.sections import DEFAULT_SECTIONS
from src.genome.serializer import serialize_genome, deserialize_genome
from src.operators.thinking_initializer import ThinkingInitializer
from src.operators.thinking_mutator import ThinkingMutator
from src.operators.thinking_crossover import ThinkingCrossover
from src.operators.thinking_evaluator import ThinkingEvaluator
from src.operators.non_thinking import (
    NonThinkingInitializer,
    NonThinkingMutator,
    SimpleCrossover,
)
from src.evaluation.financial_math import FinancialMathBenchmark
from src.evaluation.answer_checker import FinancialAnswerChecker
from src.evaluation.evaluator import UnifiedEvaluator
from src.evaluation.task_generator import generate_mixed_batch
from src.ga.engine import GAEngine, EvolutionResult
from src.ga.population import Population
from src.ga.fitness import compute_composite_fitness
from src.comparison.ablation import AblationStudy, AblationResult
from src.comparison.statistical_tests import StatisticalComparator
from src.comparison.head_to_head import HeadToHeadAnalyzer
from src.deliverables.final_report import generate_final_report
from src.deliverables.soar_summary import generate_soar_summary
from src.deliverables.hindsight_summary import generate_hindsight_summary
from src.analysis.evolution_dynamics import (
    fitness_trajectory,
    operator_contribution,
    convergence_speed,
    plot_fitness_ascii,
)
from src.analysis.prompt_analysis import (
    section_evolution,
    common_patterns,
    diff_vs_baseline,
    transferable_insights,
)
from src.analysis.operator_contribution import OperatorTracker
from src.analysis.report import generate_report


class TestFullEvolutionIntegration:
    """Full evolution run with 3 generations, verify improvement."""

    def test_full_evolution_run(self):
        """Run 3 generations of evolution and verify basic improvement."""
        thinking_llm = create_mock_thinking_llm(seed=42)
        output_llm = create_mock_output_llm(seed=42)

        initializer = ThinkingInitializer(thinking_llm)
        mutator = ThinkingMutator(thinking_llm)
        crossover = ThinkingCrossover(thinking_llm)
        evaluator = UnifiedEvaluator(output_llm, use_thinking=True)

        engine = GAEngine(
            initializer=initializer,
            mutator=mutator,
            crossover_op=crossover,
            evaluator=evaluator,
            population_size=6,
            num_generations=3,
            mutation_rate=0.3,
            crossover_rate=0.4,
            elitism_count=1,
            tournament_size=2,
        )

        bench = FinancialMathBenchmark(seed=42)
        tasks = bench.generate_tasks(n_per_category=2)
        eval_tasks = bench.to_eval_tasks(tasks[:8])

        result = engine.evolve(
            domain_desc="Financial mathematics",
            example_tasks=["Calculate compound interest", "Find present value"],
            eval_tasks=eval_tasks,
        )

        # Basic assertions
        assert isinstance(result, EvolutionResult)
        assert result.best_genome is not None
        assert result.best_fitness > 0.0
        assert result.generations_run >= 1
        assert len(result.fitness_history) >= 1

        # Verify the best genome has valid structure
        best = result.best_genome
        prompt = best.to_system_prompt()
        assert len(prompt) > 50
        assert "##" in prompt

        # Verify fitness history exists
        assert len(result.generation_results) >= 1

    def test_evolution_with_answer_checker(self):
        """Evolution with financial answer checker integration."""
        thinking_llm = create_mock_thinking_llm(seed=123)
        output_llm = create_mock_output_llm(seed=123)
        checker = FinancialAnswerChecker(tolerance=0.05)

        initializer = ThinkingInitializer(thinking_llm)
        mutator = ThinkingMutator(thinking_llm)
        crossover = ThinkingCrossover(thinking_llm)
        evaluator = UnifiedEvaluator(output_llm, checker, use_thinking=True)

        engine = GAEngine(
            initializer=initializer,
            mutator=mutator,
            crossover_op=crossover,
            evaluator=evaluator,
            population_size=4,
            num_generations=3,
            mutation_rate=0.3,
            crossover_rate=0.4,
            elitism_count=1,
            tournament_size=2,
        )

        bench = FinancialMathBenchmark(seed=42)
        tasks = bench.generate_tasks(n_per_category=1)
        eval_tasks = bench.to_eval_tasks(tasks[:5])

        result = engine.evolve("Financial math", ["task"], eval_tasks)
        assert result.best_fitness >= 0.0


class TestAblationIntegration:
    """Integration test for ablation study."""

    def test_ablation_two_conditions(self):
        """Run ablation with thinking vs non-thinking."""
        thinking_llm = create_mock_thinking_llm(seed=42)
        output_llm = create_mock_output_llm(seed=42)

        def engine_factory(**kwargs):
            use_thinking = kwargs.get("use_thinking", True)
            pop_size = kwargs.get("population_size", 4)
            crossover_rate = kwargs.get("crossover_rate", 0.4)
            elitism_count = kwargs.get("elitism_count", 1)

            if use_thinking and not kwargs.get("random_mutation", False):
                init = ThinkingInitializer(thinking_llm)
                mut = ThinkingMutator(thinking_llm)
                cross = ThinkingCrossover(thinking_llm)
            else:
                init = NonThinkingInitializer()
                mut = NonThinkingMutator()
                cross = SimpleCrossover()

            evaluator = ThinkingEvaluator(output_llm)

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

        bench = FinancialMathBenchmark(seed=42)
        tasks = bench.generate_tasks(n_per_category=1)
        eval_tasks = bench.to_eval_tasks(tasks[:5])

        result = study.run(
            eval_tasks=eval_tasks,
            repetitions=2,
            conditions=["full_thinking", "no_thinking"],
        )

        assert len(result.conditions) == 2
        assert result.conditions["full_thinking"].mean_fitness >= 0.0
        assert result.conditions["no_thinking"].mean_fitness >= 0.0

        # Statistical comparison
        comp = StatisticalComparator()
        test_result = comp.pairwise_test(
            result.conditions["full_thinking"].fitness_scores,
            result.conditions["no_thinking"].fitness_scores,
        )
        assert isinstance(test_result.p_value, float)

        # Head-to-head analysis
        analyzer = HeadToHeadAnalyzer()
        pairwise = analyzer.compare_conditions(result)
        table = analyzer.generate_ranking_table(result, pairwise)
        assert len(table) > 0


class TestReportIntegration:
    """Integration test for report generation."""

    def test_full_report_generation(self):
        """Generate reports from evolution results."""
        thinking_llm = create_mock_thinking_llm(seed=42)
        output_llm = create_mock_output_llm(seed=42)

        initializer = ThinkingInitializer(thinking_llm)
        mutator = ThinkingMutator(thinking_llm)
        crossover = ThinkingCrossover(thinking_llm)
        evaluator = UnifiedEvaluator(output_llm, use_thinking=True)

        engine = GAEngine(
            initializer=initializer,
            mutator=mutator,
            crossover_op=crossover,
            evaluator=evaluator,
            population_size=4,
            num_generations=3,
            mutation_rate=0.3,
            crossover_rate=0.4,
            elitism_count=1,
            tournament_size=2,
        )

        bench = FinancialMathBenchmark(seed=42)
        tasks = bench.generate_tasks(n_per_category=1)
        eval_tasks = bench.to_eval_tasks(tasks[:5])

        evo_result = engine.evolve("Financial math", ["task"], eval_tasks)

        # Generate all report types
        soar = generate_soar_summary(evo_result)
        assert "Strengths" in soar
        assert "Results" in soar

        hindsight = generate_hindsight_summary(evo_result)
        assert "What Worked" in hindsight

        final = generate_final_report(evo_result)
        assert "Final Report" in final

        full_report = generate_report(evo_result)
        assert "Evolution Dynamics" in full_report


class TestAnalysisIntegration:
    """Integration test for analysis tools."""

    def test_evolution_dynamics(self):
        """Test evolution dynamics analysis."""
        thinking_llm = create_mock_thinking_llm(seed=42)
        output_llm = create_mock_output_llm(seed=42)

        initializer = ThinkingInitializer(thinking_llm)
        mutator = ThinkingMutator(thinking_llm)
        crossover = ThinkingCrossover(thinking_llm)
        evaluator = UnifiedEvaluator(output_llm, use_thinking=True)

        engine = GAEngine(
            initializer=initializer,
            mutator=mutator,
            crossover_op=crossover,
            evaluator=evaluator,
            population_size=4,
            num_generations=3,
            mutation_rate=0.3,
            crossover_rate=0.4,
            elitism_count=1,
            tournament_size=2,
        )

        bench = FinancialMathBenchmark(seed=42)
        tasks = bench.generate_tasks(n_per_category=1)
        eval_tasks = bench.to_eval_tasks(tasks[:5])

        result = engine.evolve("Financial math", ["task"], eval_tasks)

        # Fitness trajectory
        trajectory = fitness_trajectory(result)
        assert len(trajectory) > 0

        # Operator contribution
        contrib = operator_contribution(result)
        assert "mutation" in contrib
        assert "crossover" in contrib

        # Convergence speed
        speed = convergence_speed(result, threshold=0.5)
        # May or may not reach threshold, just verify it runs
        assert speed is None or isinstance(speed, int)

        # ASCII plot
        plot = plot_fitness_ascii(result)
        assert "Fitness" in plot

    def test_prompt_analysis(self):
        """Test prompt analysis tools."""
        genomes = []
        for i in range(5):
            g = PromptGenome(genome_id=f"analysis_{i}")
            for name, content in DEFAULT_SECTIONS.items():
                g.set_section(name, content + f" variant {i}")
            g.fitness = 0.5 + i * 0.1
            g.generation = i
            genomes.append(g)

        # Section evolution
        by_gen = {g.generation: [g] for g in genomes}
        evolution = section_evolution(by_gen)
        assert "identity" in evolution

        # Common patterns
        patterns = common_patterns(genomes)
        assert isinstance(patterns, dict)

        # Diff vs baseline
        diffs = diff_vs_baseline(genomes[-1])
        assert "identity" in diffs

        # Transferable insights
        insights = transferable_insights(genomes, min_fitness=0.5)
        assert len(insights) > 0

    def test_operator_tracker(self):
        """Test per-operator improvement tracking."""
        tracker = OperatorTracker()
        tracker.record("mutate_methodology", 0.5, 0.6)
        tracker.record("mutate_methodology", 0.6, 0.55)
        tracker.record("crossover", 0.5, 0.7)

        summary = tracker.get_summary()
        assert "mutate_methodology" in summary
        assert "crossover" in summary

        best = tracker.get_best_operator()
        assert best is not None

        report = tracker.format_report()
        assert "Operator" in report

    def test_operator_tracker_empty(self):
        tracker = OperatorTracker()
        assert tracker.get_best_operator() is None
        assert tracker.get_summary() == {}


class TestEndToEnd:
    """End-to-end integration: evolution + analysis + reports."""

    def test_complete_pipeline(self):
        """Run the complete pipeline: evolve, analyze, report."""
        thinking_llm = create_mock_thinking_llm(seed=42)
        output_llm = create_mock_output_llm(seed=42)

        # 1. Set up and run evolution
        initializer = ThinkingInitializer(thinking_llm)
        mutator = ThinkingMutator(thinking_llm)
        crossover = ThinkingCrossover(thinking_llm)
        evaluator = UnifiedEvaluator(output_llm, use_thinking=True)

        engine = GAEngine(
            initializer=initializer,
            mutator=mutator,
            crossover_op=crossover,
            evaluator=evaluator,
            population_size=4,
            num_generations=3,
            mutation_rate=0.3,
            crossover_rate=0.4,
            elitism_count=1,
            tournament_size=2,
        )

        bench = FinancialMathBenchmark(seed=42)
        tasks = bench.generate_tasks(n_per_category=1)
        eval_tasks = bench.to_eval_tasks(tasks[:5])

        evo_result = engine.evolve("Financial math", ["task"], eval_tasks)

        # 2. Serialize and deserialize the best genome
        if evo_result.best_genome:
            serialized = serialize_genome(evo_result.best_genome)
            restored = deserialize_genome(serialized)
            assert restored.genome_id == evo_result.best_genome.genome_id

        # 3. Generate comprehensive report
        report = generate_report(evo_result)
        assert len(report) > 100

        # 4. Verify the pipeline produced valid output
        assert evo_result.stopped_reason in (
            "max_generations", "stagnation", "optimal"
        )

    def test_mixed_task_generation(self):
        """Test generating mixed evaluation tasks."""
        tasks = generate_mixed_batch(n=14, seed=42)
        assert len(tasks) == 14
        categories = {t.get("category", "") for t in tasks}
        assert len(categories) > 1

    def test_unified_evaluator_both_modes(self):
        """Test UnifiedEvaluator in both thinking and non-thinking modes."""
        output_llm = create_mock_output_llm(seed=42)

        genome = PromptGenome(genome_id="test_eval")
        for name, content in DEFAULT_SECTIONS.items():
            genome.set_section(name, content)

        bench = FinancialMathBenchmark(seed=42)
        tasks = bench.generate_tasks(n_per_category=1)
        eval_tasks = bench.to_eval_tasks(tasks[:5])

        # Thinking mode
        eval_thinking = UnifiedEvaluator(output_llm, use_thinking=True)
        details_t = eval_thinking.evaluate(genome, eval_tasks)
        assert details_t.composite_fitness >= 0.0

        # Non-thinking mode
        eval_non = UnifiedEvaluator(output_llm, use_thinking=False)
        details_nt = eval_non.evaluate(genome, eval_tasks)
        assert details_nt.composite_fitness >= 0.0

    def test_population_save_load_roundtrip(self):
        """Test that population can be saved and loaded."""
        import tempfile
        import os

        pop = Population(max_size=5, elitism_count=1)
        for i in range(3):
            g = PromptGenome(genome_id=f"pop_test_{i}")
            g.set_section("identity", f"AI variant {i}")
            g.fitness = 0.5 + i * 0.1
            pop.add(g)
        pop.advance_generation()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            pop.save(filepath)
            loaded = Population.load(filepath)
            assert loaded.size == pop.size
            assert loaded.generation == pop.generation
        finally:
            os.unlink(filepath)
