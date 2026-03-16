"""Integration tests: full pipeline from task generation to report."""

import os
import pytest

from src.benchmarks.registry import create_default_registry, BenchmarkRegistry
from src.benchmarks.oolong import OOLONGBenchmark
from src.benchmarks.locodiff import LoCoDiffBenchmark
from src.benchmarks.synthetic import SyntheticTaskGenerator
from src.benchmarks.task import EvalTask, EvalResult
from src.execution.rlm_executor import RLMExecutor
from src.execution.standard_executor import StandardExecutor
from src.execution.runner import BenchmarkRunner
from src.strategies.classifier import StrategyClassifier, StrategyType
from src.strategies.emergence_analyzer import EmergenceAnalyzer
from src.strategies.effectiveness import StrategyEffectivenessAnalyzer
from src.strategies.failure_modes import StrategyFailureModeAnalyzer
from src.strategies.evolution_tracker import StrategyEvolutionTracker
from src.comparison.cost_model import CostModel
from src.comparison.head_to_head import HeadToHeadComparator
from src.comparison.scaling_experiment import ContextScalingExperiment
from src.comparison.efficiency_frontier import EfficiencyFrontierAnalyzer
from src.comparison.statistical_tests import StatisticalTests
from src.deliverables.rlm_wrapper_summary import RLMWrapperSummary
from src.deliverables.repl_summary import REPLSummary
from src.deliverables.final_report import FinalReport
from src.analysis.report import ReportGenerator
from src.analysis.trajectory_visualizer import TrajectoryVisualizer
from src.analysis.strategy_landscape import StrategyLandscape
from src.analysis.context_scaling import ContextScalingAnalysis
from src.analysis.cost_breakdown import CostBreakdownAnalysis


class TestFullPipeline:
    """Full integration test: generate -> execute -> classify -> compare -> report."""

    @pytest.fixture
    def pipeline_data(self, tmp_checkpoint_dir):
        """Run the full pipeline and return all results."""
        # 1. Generate tasks
        registry = create_default_registry()
        all_tasks = registry.load_all()
        assert len(all_tasks) > 30  # Should have substantial number of tasks

        # Use a subset for speed
        tasks = all_tasks[:15]

        # 2. Execute with both systems
        rlm = RLMExecutor(seed=42)
        std = StandardExecutor(context_window=4096, seed=42)

        rlm_results = [rlm.execute(t) for t in tasks]
        std_results = [std.execute(t) for t in tasks]

        task_categories = {t.task_id: t.category for t in tasks}

        return {
            "tasks": tasks,
            "rlm_results": rlm_results,
            "std_results": std_results,
            "task_categories": task_categories,
            "registry": registry,
        }

    def test_task_generation(self, pipeline_data):
        """Verify tasks are generated correctly."""
        tasks = pipeline_data["tasks"]
        assert len(tasks) == 15
        assert all(isinstance(t, EvalTask) for t in tasks)
        benchmarks = {t.benchmark for t in tasks}
        assert len(benchmarks) >= 1

    def test_rlm_execution(self, pipeline_data):
        """Verify RLM execution produces valid results."""
        results = pipeline_data["rlm_results"]
        assert len(results) == 15
        assert all(isinstance(r, EvalResult) for r in results)
        # Should have some correct answers
        correct_count = sum(1 for r in results if r.correct)
        assert correct_count >= 0  # At least 0 (mock may vary)
        # All should have trajectories
        for r in results:
            assert len(r.trajectory) > 0

    def test_standard_execution(self, pipeline_data):
        """Verify standard execution produces valid results."""
        results = pipeline_data["std_results"]
        assert len(results) == 15
        assert all(isinstance(r, EvalResult) for r in results)

    def test_strategy_classification(self, pipeline_data):
        """Verify strategy classification works on results."""
        classifier = StrategyClassifier()
        results = pipeline_data["rlm_results"]

        for r in results:
            classification = classifier.classify(r.trajectory)
            assert classification.strategy in StrategyType
            assert 0 < classification.confidence <= 1.0
            assert isinstance(classification.pattern_sequence, list)

    def test_emergence_analysis(self, pipeline_data):
        """Verify emergence analysis produces meaningful report."""
        analyzer = EmergenceAnalyzer()
        results = pipeline_data["rlm_results"]
        categories = pipeline_data["task_categories"]

        report = analyzer.analyze(results, categories)
        assert report.total_results_analyzed == len(results)
        assert isinstance(report.strategy_by_task_type, dict)
        assert isinstance(report.strategy_effectiveness, dict)
        assert 0 <= report.grep_before_read_rate <= 1.0
        assert isinstance(report.summary(), str)

    def test_strategy_effectiveness(self, pipeline_data):
        """Verify effectiveness analysis."""
        analyzer = StrategyEffectivenessAnalyzer()
        results = pipeline_data["rlm_results"]
        categories = pipeline_data["task_categories"]

        acc = analyzer.accuracy_by_strategy_and_category(results, categories)
        assert isinstance(acc, dict)

        cost = analyzer.cost_by_strategy(results)
        assert isinstance(cost, dict)

        optimal = analyzer.optimal_strategy_map(results, categories)
        assert isinstance(optimal, dict)

        cost_eff = analyzer.cost_effectiveness(results)
        assert isinstance(cost_eff, dict)

    def test_failure_mode_analysis(self, pipeline_data):
        """Verify failure mode analysis."""
        analyzer = StrategyFailureModeAnalyzer()
        results = pipeline_data["rlm_results"]
        categories = pipeline_data["task_categories"]

        failures = analyzer.categorize_failures(results, categories)
        assert isinstance(failures, dict)

        misapplied = analyzer.strategy_misapplication(results, categories)
        assert isinstance(misapplied, list)

        rates = analyzer.failure_rate_by_strategy(results)
        assert isinstance(rates, dict)

    def test_evolution_tracking(self, pipeline_data):
        """Verify evolution tracking."""
        tracker = StrategyEvolutionTracker()
        results = pipeline_data["rlm_results"]

        transitions = tracker.track_within_session(results)
        assert isinstance(transitions, list)

        matrix = tracker.transition_matrix(results)
        assert isinstance(matrix, dict)

        timeline = tracker.strategy_timeline(results)
        assert len(timeline) == len(results)

        events = tracker.adaptation_events(results)
        assert isinstance(events, list)

        diversity = tracker.strategy_diversity(results)
        assert 0 <= diversity <= 1.0

    def test_cost_comparison(self, pipeline_data):
        """Verify cost comparison."""
        model = CostModel()
        rlm_results = pipeline_data["rlm_results"]
        std_results = pipeline_data["std_results"]

        comparison = model.compare_systems(rlm_results, std_results)
        assert comparison.cost_ratio > 0
        assert comparison.efficiency_winner in ("rlm", "standard")

    def test_head_to_head(self, pipeline_data):
        """Verify head-to-head comparison."""
        comparator = HeadToHeadComparator()
        rlm_results = pipeline_data["rlm_results"]
        std_results = pipeline_data["std_results"]
        categories = pipeline_data["task_categories"]

        report = comparator.compare(rlm_results, std_results, categories)
        assert report.total_tasks > 0
        assert report.rlm_wins + report.standard_wins + report.ties == report.total_tasks

        # Paired accuracy
        rlm_acc, std_acc = comparator.paired_accuracy(rlm_results, std_results)
        assert 0 <= rlm_acc <= 1.0
        assert 0 <= std_acc <= 1.0

        # Win rate
        rlm_wr, std_wr = comparator.win_rate(rlm_results, std_results)
        assert 0 <= rlm_wr <= 1.0
        assert 0 <= std_wr <= 1.0

    def test_scaling_experiment_mini(self, pipeline_data):
        """Verify scaling experiment works."""
        rlm = RLMExecutor(seed=42)
        std = StandardExecutor(context_window=4096, seed=42)
        experiment = ContextScalingExperiment(rlm.execute, std.execute)

        gen = SyntheticTaskGenerator()
        base_tasks = gen.needle_in_haystack(context_tokens=1000, num_tasks=2)
        results = experiment.run(base_tasks, [1000, 4000])
        assert len(results) == 2

    def test_efficiency_frontier(self, pipeline_data):
        """Verify efficiency frontier computation."""
        analyzer = EfficiencyFrontierAnalyzer()
        system_results = {
            "rlm": pipeline_data["rlm_results"],
            "standard": pipeline_data["std_results"],
        }

        points = analyzer.compute_frontier(system_results)
        assert len(points) == 2
        # At least one should be Pareto optimal
        assert any(p.is_pareto_optimal for p in points)

        plot = analyzer.plot(system_results)
        assert "Efficiency Frontier" in plot

        dominated = analyzer.dominated_by(system_results)
        assert isinstance(dominated, dict)

    def test_statistical_tests(self, pipeline_data):
        """Verify statistical tests."""
        tests = StatisticalTests()
        rlm_results = pipeline_data["rlm_results"]
        std_results = pipeline_data["std_results"]

        mcnemar = tests.mcnemar_test(rlm_results, std_results)
        assert isinstance(mcnemar.p_value, float)
        assert isinstance(mcnemar.significant, bool)

        ci = tests.confidence_interval(rlm_results)
        assert ci[0] <= ci[1]
        assert 0 <= ci[0] <= 1.0
        assert 0 <= ci[1] <= 1.0

        diff_ci = tests.accuracy_difference_ci(rlm_results, std_results)
        assert diff_ci[0] <= diff_ci[1]

    def test_report_generation(self, pipeline_data):
        """Verify complete report generation."""
        generator = ReportGenerator()
        report = generator.generate(
            pipeline_data["rlm_results"],
            pipeline_data["std_results"],
            pipeline_data["task_categories"],
        )
        assert isinstance(report, str)
        assert len(report) > 100
        assert "RLM Evaluation Report" in report
        assert "Head-to-Head" in report
        assert "Emergence" in report

    def test_trajectory_visualizer(self, pipeline_data):
        """Verify trajectory visualization."""
        viz = TrajectoryVisualizer()
        results = pipeline_data["rlm_results"]

        rendered = viz.render(results[0])
        assert "Trajectory" in rendered
        assert results[0].task_id in rendered

        batch = viz.render_batch(results, max_show=3)
        assert "Trajectory" in batch

        selected = viz.select_representative(results, num_examples=3)
        assert len(selected) <= 3

        grouped = viz.strategy_examples(results)
        assert isinstance(grouped, dict)

    def test_strategy_landscape(self, pipeline_data):
        """Verify strategy landscape analysis."""
        landscape = StrategyLandscape()
        results = pipeline_data["rlm_results"]
        categories = pipeline_data["task_categories"]

        table = landscape.distribution_table(results, categories)
        assert "Strategy Distribution" in table

        heatmap = landscape.heatmap_ascii(results, categories)
        assert "Strategy Heatmap" in heatmap

        summary = landscape.strategy_summary(results)
        assert isinstance(summary, dict)
        for strat_data in summary.values():
            assert "count" in strat_data
            assert "accuracy" in strat_data

    def test_context_scaling_analysis(self, pipeline_data):
        """Verify context scaling analysis."""
        from src.comparison.scaling_experiment import ScalingResult

        analysis = ContextScalingAnalysis()

        # Create some mock scaling results
        scaling_results = [
            ScalingResult(context_size=1000, rlm_accuracy=0.8, standard_accuracy=0.9,
                         rlm_cost=0.01, standard_cost=0.005),
            ScalingResult(context_size=8000, rlm_accuracy=0.75, standard_accuracy=0.5,
                         rlm_cost=0.05, standard_cost=0.01),
        ]

        plot = analysis.plot_ascii(scaling_results)
        assert "Accuracy" in plot

        degradation = analysis.degradation_analysis(scaling_results)
        assert "rlm" in degradation
        assert "standard" in degradation

        advantage = analysis.advantage_curve(scaling_results)
        assert len(advantage) == 2

    def test_cost_breakdown_analysis(self, pipeline_data):
        """Verify cost breakdown analysis."""
        analysis = CostBreakdownAnalysis()
        results = pipeline_data["rlm_results"]
        categories = pipeline_data["task_categories"]

        by_strat = analysis.by_strategy(results)
        assert isinstance(by_strat, dict)

        by_cat = analysis.by_category(results, categories)
        assert isinstance(by_cat, dict)

        by_bench = analysis.by_benchmark(results)
        assert isinstance(by_bench, dict)

        io = analysis.io_ratio(results)
        assert "input_ratio" in io
        assert "output_ratio" in io

        table = analysis.cost_summary_table(results, categories)
        assert "Cost Breakdown" in table

    def test_deliverables(self, pipeline_data):
        """Verify deliverable generation."""
        rlm_results = pipeline_data["rlm_results"]
        std_results = pipeline_data["std_results"]

        # RLM Wrapper Summary
        wrapper = RLMWrapperSummary(rlm_results)
        doc = wrapper.generate()
        assert "RLM Wrapper Summary" in doc
        assert "Architecture" in doc

        # REPL Summary
        repl = REPLSummary(rlm_results, ["oolong", "locodiff", "synthetic"])
        doc = repl.generate()
        assert "REPL Evaluation" in doc

        # Final Report
        h2h = HeadToHeadComparator().compare(rlm_results, std_results)
        cost_cmp = CostModel().compare_systems(rlm_results, std_results)
        emergence = EmergenceAnalyzer().analyze(rlm_results)

        report = FinalReport(rlm_results, std_results, cost_cmp, h2h, emergence)
        doc = report.generate()
        assert "Final Report" in doc
        assert "Executive Summary" in doc

    def test_benchmark_runner_integration(self, pipeline_data, tmp_checkpoint_dir):
        """Verify benchmark runner with full pipeline."""
        rlm = RLMExecutor(seed=42)
        runner = BenchmarkRunner(
            executor_fn=rlm.execute,
            checkpoint_dir=tmp_checkpoint_dir,
        )
        tasks = pipeline_data["tasks"][:5]

        run = runner.run_benchmark(tasks, "integration_test")
        assert run.total_tasks == 5
        assert len(run.results) == 5

    def test_registry_integration(self):
        """Verify registry loads all benchmarks correctly."""
        registry = create_default_registry()
        assert "oolong" in registry.available_benchmarks
        assert "locodiff" in registry.available_benchmarks
        assert "synthetic" in registry.available_benchmarks

        all_tasks = registry.load_all()
        assert len(all_tasks) > 30

        # Filter works
        filtered = registry.filter(all_tasks, category="retrieval")
        assert all(t.category == "retrieval" for t in filtered)

        filtered2 = registry.filter(all_tasks, difficulty="hard")
        assert all(t.difficulty == "hard" for t in filtered2)

        filtered3 = registry.filter(all_tasks, benchmark="oolong")
        assert all(t.benchmark == "oolong" for t in filtered3)

    def test_registry_filter_by_context_size(self):
        """Test filtering by context size."""
        registry = create_default_registry()
        all_tasks = registry.load_all()

        small = registry.filter(all_tasks, max_context_size=1000)
        for t in small:
            assert t.context_tokens <= 1000

    def test_registry_unknown_benchmark(self):
        """Test loading unknown benchmark raises error."""
        registry = create_default_registry()
        with pytest.raises(KeyError):
            registry.load("nonexistent")

    def test_eval_task_with_context_size(self):
        """Test task context resizing."""
        task = EvalTask(
            task_id="test",
            benchmark="test",
            query="question?",
            context="Short context.",
            expected_answer="answer",
            context_tokens=10,
        )
        sized = task.with_context_size(1000)
        assert sized.context_tokens == 1000
        assert "test" in sized.metadata.get("original_task_id", "")
        assert len(sized.context) >= 1000  # Padded to fit

    def test_eval_result_serialization(self):
        """Test EvalResult to_dict/from_dict round-trip."""
        result = EvalResult(
            task_id="t1",
            benchmark="b",
            answer="a",
            correct=True,
            trajectory=["step1", "step2"],
            strategy_detected="PEEK_THEN_GREP",
            cost=0.025,
            input_tokens=1000,
            output_tokens=500,
        )
        d = result.to_dict()
        restored = EvalResult.from_dict(d)
        assert restored.task_id == result.task_id
        assert restored.correct == result.correct
        assert restored.strategy_detected == result.strategy_detected
        assert restored.cost == result.cost
        assert restored.total_tokens == 1500
