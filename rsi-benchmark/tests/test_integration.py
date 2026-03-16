"""Integration tests: end-to-end pipeline, improvement curves, collapse, ablation, report."""

import pytest

from tests.conftest import MockAgent, MockPipeline
from src.benchmarks.registry import BenchmarkRegistry, register_all_benchmarks
from src.evaluation.iteration_evaluator import IterationEvaluator, IterationEvaluation
from src.evaluation.improvement_curves import ImprovementCurveTracker
from src.evaluation.held_out_evaluator import HeldOutEvaluator
from src.evaluation.snapshot_evaluator import SnapshotEvaluator, AgentSnapshot
from src.collapse.baseline_loader import CollapseBaselineLoader
from src.collapse.divergence_analyzer import DivergenceAnalyzer
from src.collapse.entropy_tracker import EntropyTracker
from src.collapse.sustainability import SustainabilityAnalyzer
from src.ablation.ablation_study import ParadigmAblationStudy
from src.ablation.contribution import ContributionAnalyzer
from src.ablation.conditions import build_all_conditions
from src.analysis.cross_benchmark import (
    correlation_matrix,
    transfer_effects,
    benchmark_difficulty_ranking,
)
from src.analysis.scaling_analysis import fit_scaling_law, optimal_iterations, project_forward
from src.analysis.cost_analysis import cost_breakdown, cost_per_improvement_point, format_cost_table
from src.analysis.qualitative import select_examples, annotate
from src.analysis.report import generate_report
from src.deliverables.pipeline_summary import PipelineSummary
from src.deliverables.benchmark_summary import BenchmarkSummary
from src.deliverables.ablation_summary import AblationSummary
from src.deliverables.final_report import FinalReport


class TestEndToEndPipeline:
    """Test the full pipeline: 3 iterations on 2 benchmarks."""

    def test_full_pipeline_run(self, sample_benchmarks):
        agent = MockAgent("full_pipeline")
        evaluator = IterationEvaluator(sample_benchmarks)
        tracker = ImprovementCurveTracker()

        evaluations = []
        for iteration in range(3):
            agent.set_iteration(iteration)
            evaluation = evaluator.evaluate_iteration(agent, iteration)
            evaluations.append(evaluation)

            assert isinstance(evaluation, IterationEvaluation)
            assert evaluation.iteration == iteration
            assert len(evaluation.accuracy_by_benchmark) == 2

            for bm_name, acc in evaluation.accuracy_by_benchmark.items():
                tracker.record(bm_name, iteration, acc)
                assert 0.0 <= acc <= 1.0

        # Check improvement
        for bm_name in sample_benchmarks:
            total = tracker.compute_total_improvement(bm_name)
            # Should show some improvement over 3 iterations
            assert total >= -0.1  # Allow small fluctuation

    def test_consistent_task_sets(self, sample_benchmarks):
        """Verify same tasks are used across iterations."""
        evaluator = IterationEvaluator(sample_benchmarks)
        tasks = evaluator.task_sets

        for bm_name in sample_benchmarks:
            assert bm_name in tasks
            assert len(tasks[bm_name]) > 0

    def test_category_accuracy(self, sample_benchmarks):
        agent = MockAgent("full_pipeline")
        evaluator = IterationEvaluator(sample_benchmarks)
        evaluation = evaluator.evaluate_iteration(agent, 0)
        assert "math500" in evaluation.category_accuracy
        cats = evaluation.category_accuracy["math500"]
        assert len(cats) > 0


class TestImprovementCurvesIntegration:
    """Test improvement curves with real benchmark evaluations."""

    def test_improvement_over_iterations(self, sample_benchmarks):
        agent = MockAgent("full_pipeline")
        evaluator = IterationEvaluator(sample_benchmarks)
        tracker = ImprovementCurveTracker()

        for iteration in range(3):
            agent.set_iteration(iteration)
            evaluation = evaluator.evaluate_iteration(agent, iteration)
            for bm_name, acc in evaluation.accuracy_by_benchmark.items():
                tracker.record(bm_name, iteration, acc)

        # Get curves
        for bm_name in sample_benchmarks:
            curve = tracker.get_curve(bm_name)
            assert len(curve) == 3

    def test_growth_model_fitting_integration(self, sample_benchmarks):
        agent = MockAgent("full_pipeline")
        evaluator = IterationEvaluator(sample_benchmarks)
        tracker = ImprovementCurveTracker()

        for iteration in range(1, 4):  # Start from 1 for log
            agent.set_iteration(iteration)
            evaluation = evaluator.evaluate_iteration(agent, iteration)
            for bm_name, acc in evaluation.accuracy_by_benchmark.items():
                tracker.record(bm_name, iteration, acc)

        for bm_name in sample_benchmarks:
            model = tracker.fit_growth_model(bm_name, "logarithmic")
            assert model.model_type == "logarithmic"


class TestCollapseComparisonIntegration:
    """Test collapse comparison with real evaluation data."""

    def test_rsi_vs_collapse(self, sample_benchmarks, collapse_baselines):
        agent = MockAgent("full_pipeline")
        evaluator = IterationEvaluator(sample_benchmarks)
        tracker = ImprovementCurveTracker()

        for iteration in range(3):
            agent.set_iteration(iteration)
            evaluation = evaluator.evaluate_iteration(agent, iteration)
            for bm_name, acc in evaluation.accuracy_by_benchmark.items():
                tracker.record(bm_name, iteration, acc)

        collapse = collapse_baselines.load("standard_decay")
        analyzer = DivergenceAnalyzer()

        for bm_name in sample_benchmarks:
            curve = tracker.get_curve(bm_name)
            result = analyzer.compute_divergence(curve, collapse.accuracy)
            assert len(result.divergence_values) > 0

    def test_entropy_tracking_integration(self, sample_benchmarks):
        entropy_tracker = EntropyTracker()
        agent = MockAgent("full_pipeline")

        for iteration in range(3):
            agent.set_iteration(iteration)
            outputs = []
            for bm_name, bm in sample_benchmarks.items():
                for task in bm.tasks[:5]:
                    answer = agent.solve(task)
                    outputs.append(str(answer))
            entropy_tracker.record_outputs(iteration, outputs)

        curve = entropy_tracker.compute_entropy_curve()
        assert len(curve) == 3

    def test_sustainability_integration(self, sample_benchmarks):
        agent = MockAgent("full_pipeline")
        evaluator = IterationEvaluator(sample_benchmarks)

        accuracies = []
        for iteration in range(3):
            agent.set_iteration(iteration)
            evaluation = evaluator.evaluate_iteration(agent, iteration)
            accuracies.append(evaluation.overall_accuracy)

        analyzer = SustainabilityAnalyzer()
        report = analyzer.analyze(accuracies)
        assert report.overall_sustainability_score >= 0.0


class TestAblationIntegration:
    """Test mini ablation study."""

    def test_mini_ablation(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=3,
        )
        result = study.run(sample_benchmarks)

        assert len(result.conditions) == 7
        assert len(result.benchmarks) == 2

        # Verify naive declines
        assert result.summary["naive_self_train"] < result.summary["full_pipeline"]

    def test_ablation_contributions(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=5,
        )
        result = study.run(sample_benchmarks)

        analyzer = ContributionAnalyzer()
        contributions = analyzer.compute_contributions(result)
        assert len(contributions) == 4

        ranked = analyzer.rank_paradigms(result)
        assert ranked[0].rank == 1


class TestCrossBenchmarkAnalysis:
    """Test cross-benchmark analysis."""

    def test_correlation_matrix(self):
        data = {
            "math500": [0.60, 0.62, 0.64, 0.66],
            "humaneval": [0.58, 0.61, 0.63, 0.65],
        }
        matrix = correlation_matrix(data)
        assert "math500" in matrix
        assert "humaneval" in matrix
        # Self-correlation should be ~1
        assert abs(matrix["math500"]["math500"] - 1.0) < 0.01
        # Positive correlation expected
        assert matrix["math500"]["humaneval"] > 0.5

    def test_transfer_effects(self):
        data = {
            "math500": [0.60, 0.62, 0.64],
            "humaneval": [0.58, 0.61, 0.63],
        }
        effects = transfer_effects(data)
        assert "math500" in effects
        assert effects["math500"]["math500"] == 1.0

    def test_benchmark_difficulty_ranking(self):
        accuracies = {"math500": 0.75, "arc_agi": 0.65, "humaneval": 0.80}
        ranking = benchmark_difficulty_ranking(accuracies)
        assert len(ranking) == 3
        # Hardest first
        assert ranking[0][0] == "arc_agi"
        assert ranking[0][2] == 1  # Rank 1 (hardest)


class TestScalingAnalysis:
    """Test scaling analysis."""

    def test_fit_scaling_law_linear(self):
        iterations = list(range(1, 11))
        accuracies = [0.60 + 0.015 * i for i in iterations]
        law = fit_scaling_law(iterations, accuracies, "linear")
        assert law.law_type == "linear"
        assert law.r_squared > 0.9
        assert "a" in law.params
        assert "b" in law.params

    def test_fit_scaling_law_logarithmic(self):
        import math as m
        iterations = list(range(1, 11))
        accuracies = [0.5 + 0.1 * m.log(i) for i in iterations]
        law = fit_scaling_law(iterations, accuracies, "logarithmic")
        assert law.law_type == "logarithmic"
        assert law.r_squared > 0.9

    def test_fit_scaling_law_power(self):
        iterations = list(range(1, 11))
        accuracies = [0.5 + 0.05 * (i ** 0.5) for i in iterations]
        law = fit_scaling_law(iterations, accuracies, "power")
        assert law.law_type == "power"
        assert "a" in law.params
        assert "b" in law.params
        assert "c" in law.params

    def test_fit_scaling_law_empty(self):
        law = fit_scaling_law([], [], "linear")
        assert law.r_squared == 0.0
        assert law.domain == (0, 0)

    def test_fit_scaling_law_unknown_raises(self):
        with pytest.raises(ValueError):
            fit_scaling_law([1, 2], [0.5, 0.6], "cubic")

    def test_scaling_law_predict_linear(self):
        from src.analysis.scaling_analysis import ScalingLaw
        law = ScalingLaw(law_type="linear", params={"a": 0.01, "b": 0.5}, r_squared=0.99, domain=(1, 10))
        assert abs(law.predict(10) - 0.6) < 0.01

    def test_scaling_law_predict_logarithmic(self):
        import math as m
        from src.analysis.scaling_analysis import ScalingLaw
        law = ScalingLaw(law_type="logarithmic", params={"a": 0.1, "b": 0.5}, r_squared=0.99, domain=(1, 10))
        pred = law.predict(m.e)
        assert abs(pred - 0.6) < 0.01

    def test_scaling_law_predict_power(self):
        from src.analysis.scaling_analysis import ScalingLaw
        law = ScalingLaw(law_type="power", params={"a": 1.0, "b": 0.5, "c": 0.0}, r_squared=0.99, domain=(1, 10))
        pred = law.predict(4.0)
        assert abs(pred - 2.0) < 0.01

    def test_scaling_law_predict_unknown(self):
        from src.analysis.scaling_analysis import ScalingLaw
        law = ScalingLaw(law_type="unknown", params={}, r_squared=0.0, domain=(0, 0))
        assert law.predict(5.0) == 0.0

    def test_optimal_iterations(self):
        iterations = list(range(1, 11))
        accuracies = [0.60 + 0.015 * i for i in iterations]
        law = fit_scaling_law(iterations, accuracies, "linear")
        opt = optimal_iterations(law, target_accuracy=0.70)
        assert opt is not None
        assert opt > 0

    def test_optimal_iterations_unreachable(self):
        iterations = list(range(1, 11))
        accuracies = [0.60 + 0.001 * i for i in iterations]
        law = fit_scaling_law(iterations, accuracies, "linear")
        opt = optimal_iterations(law, target_accuracy=0.99, max_iterations=20)
        assert opt is None

    def test_project_forward(self):
        iterations = list(range(1, 11))
        accuracies = [0.60 + 0.015 * i for i in iterations]
        law = fit_scaling_law(iterations, accuracies, "linear")
        projections = project_forward(law, 5)
        assert len(projections) == 5
        # Projected values should be increasing
        for i in range(len(projections) - 1):
            assert projections[i + 1][1] >= projections[i][1]

    def test_project_forward_logarithmic(self):
        import math as m
        iterations = list(range(1, 11))
        accuracies = [0.5 + 0.1 * m.log(i) for i in iterations]
        law = fit_scaling_law(iterations, accuracies, "logarithmic")
        projections = project_forward(law, 3)
        assert len(projections) == 3
        assert projections[0][0] == 11


class TestCostAnalysis:
    """Test cost analysis."""

    def test_cost_breakdown(self):
        cb = cost_breakdown(
            iterations=15,
            benchmarks=["math500", "humaneval"],
            cost_per_task=0.01,
            tasks_per_benchmark=30,
            improvement=0.2,
        )
        assert cb.total_cost > 0
        assert cb.cost_per_iteration > 0
        assert len(cb.cost_per_benchmark) == 2
        assert cb.cost_per_improvement_point > 0

    def test_cost_per_improvement_point(self):
        assert cost_per_improvement_point(100, 0.1) == 1000
        assert cost_per_improvement_point(100, 0) == float("inf")

    def test_format_cost_table(self):
        cb = cost_breakdown(10, ["math500"], improvement=0.1)
        table = format_cost_table(cb)
        assert "Total Cost" in table
        assert "math500" in table


class TestQualitativeAnalysis:
    """Test qualitative analysis."""

    def test_select_examples(self, sample_benchmarks):
        from src.benchmarks.registry import BenchmarkResult
        agent0 = MockAgent("full_pipeline")
        agent0.set_iteration(0)
        agent10 = MockAgent("full_pipeline")
        agent10.set_iteration(10)

        bm = list(sample_benchmarks.values())[0]
        results_0 = bm.evaluate(agent0)
        results_10 = bm.evaluate(agent10)

        results_by_iteration = {0: results_0, 10: results_10}
        examples = select_examples(results_by_iteration, max_examples=5)
        assert len(examples) <= 5

    def test_annotate_example(self):
        from src.analysis.qualitative import AnnotatedExample
        ex = AnnotatedExample(
            task_id="test_01",
            benchmark="math500",
            iteration=5,
            category="improvement",
            predicted=42,
            expected=42,
            correct=True,
            annotation="Initial",
        )
        annotated = annotate(ex, "This is a notable improvement", ["notable"])
        assert annotated.annotation == "This is a notable improvement"
        assert "notable" in annotated.tags


class TestHeldOutEvaluator:
    """Test held-out evaluation."""

    def test_held_out_split(self, sample_benchmarks):
        held_out_eval = HeldOutEvaluator(sample_benchmarks, held_out_fraction=0.2)
        for bm_name in sample_benchmarks:
            assert len(held_out_eval.held_out_tasks[bm_name]) >= 1
            assert len(held_out_eval.training_tasks[bm_name]) >= 1

    def test_held_out_evaluate(self, sample_benchmarks):
        held_out_eval = HeldOutEvaluator(sample_benchmarks, held_out_fraction=0.2)
        agent = MockAgent("full_pipeline")
        results = held_out_eval.evaluate(agent)
        for bm_name in sample_benchmarks:
            assert bm_name in results
            assert 0.0 <= results[bm_name].accuracy <= 1.0

    def test_generalization_gap(self, sample_benchmarks):
        held_out_eval = HeldOutEvaluator(sample_benchmarks, held_out_fraction=0.2)
        agent = MockAgent("full_pipeline")
        training_acc = {"math500": 0.70, "humaneval": 0.65}
        results = held_out_eval.evaluate(agent, training_accuracy=training_acc)
        for bm_name in sample_benchmarks:
            # generalization_gap can be positive or negative
            assert isinstance(results[bm_name].generalization_gap, float)


class TestSnapshotEvaluator:
    """Test snapshot evaluation."""

    def test_evaluate_snapshot(self, sample_benchmarks):
        agent = MockAgent("full_pipeline")
        agent.set_iteration(5)
        snapshot = AgentSnapshot(
            snapshot_id="snap_5",
            iteration=5,
            agent_state=agent,
        )
        evaluator = SnapshotEvaluator(sample_benchmarks)
        result = evaluator.evaluate(snapshot)
        assert result.snapshot_id == "snap_5"
        assert result.iteration == 5
        assert 0.0 <= result.overall_accuracy <= 1.0

    def test_evaluate_multiple_snapshots(self, sample_benchmarks):
        snapshots = []
        for i in [0, 5, 10]:
            agent = MockAgent("full_pipeline")
            agent.set_iteration(i)
            snapshots.append(AgentSnapshot(
                snapshot_id=f"snap_{i}",
                iteration=i,
                agent_state=agent,
            ))

        evaluator = SnapshotEvaluator(sample_benchmarks)
        results = evaluator.evaluate_all(snapshots)
        assert len(results) == 3

    def test_compare_snapshots(self, sample_benchmarks):
        snapshots = []
        for i in [0, 5, 10]:
            agent = MockAgent("full_pipeline")
            agent.set_iteration(i)
            snapshots.append(AgentSnapshot(
                snapshot_id=f"snap_{i}",
                iteration=i,
                agent_state=agent,
            ))

        evaluator = SnapshotEvaluator(sample_benchmarks)
        results = evaluator.evaluate_all(snapshots)
        comparison = evaluator.compare_snapshots(results)

        assert "iterations" in comparison
        assert "overall_accuracy" in comparison
        assert "total_improvement" in comparison
        assert comparison["total_improvement"] != 0  # Should show change

    def test_compare_empty(self, sample_benchmarks):
        evaluator = SnapshotEvaluator(sample_benchmarks)
        assert evaluator.compare_snapshots([]) == {}


class TestReportGeneration:
    """Test report generation."""

    def test_generate_full_report(self):
        report = generate_report(
            pipeline_config={"iterations": 3, "benchmarks": ["math500", "humaneval"]},
            benchmark_results={
                "math500": {
                    "final_accuracy": 0.75,
                    "improvement": 0.15,
                    "num_tasks": 32,
                    "categories": ["algebra", "calculus"],
                },
                "humaneval": {
                    "final_accuracy": 0.70,
                    "improvement": 0.10,
                    "num_tasks": 15,
                    "categories": ["function_completion"],
                },
            },
            improvement_data={
                "overall_improvement": 0.125,
                "final_accuracy": 0.725,
                "curves": {"math500": [0.60, 0.68, 0.75], "humaneval": [0.60, 0.65, 0.70]},
                "growth_models": {"math500": {"type": "log", "r_squared": 0.95}},
            },
            collapse_data={"prevention_score": 0.85, "sustainability_score": 0.90},
        )

        assert "# RSI Benchmark Report" in report
        assert "Executive Summary" in report
        assert "Methodology" in report
        assert "Pipeline Configuration" in report
        assert "Benchmark Descriptions" in report
        assert "Iteration Results" in report
        assert "Improvement Curves" in report
        assert "Growth Model" in report
        assert "Collapse Comparison" in report
        assert "Sustainability" in report
        assert "Ablation" in report
        assert "Cross-Benchmark" in report
        assert "Scaling" in report
        assert "Cost" in report
        assert "Conclusions" in report

    def test_report_has_14_sections(self):
        report = generate_report()
        # Count major section headers
        section_count = report.count("## ")
        assert section_count >= 14


class TestDeliverablePackaging:
    """Test deliverable packaging."""

    def test_pipeline_summary(self):
        summary = PipelineSummary()
        result = summary.generate(
            iterations=15,
            benchmarks=["math500", "humaneval"],
            overall_improvement=0.20,
            final_accuracy=0.80,
        )
        assert result["pipeline"]["total_iterations"] == 15
        assert result["results"]["overall_improvement"] == 0.20

    def test_pipeline_summary_markdown(self):
        summary = PipelineSummary()
        md = summary.to_markdown(
            iterations=15,
            benchmarks=["math500"],
            overall_improvement=0.20,
            final_accuracy=0.80,
        )
        assert "Pipeline Summary" in md
        assert "15" in md

    def test_pipeline_summary_record(self):
        summary = PipelineSummary()
        summary.record("custom_key", "custom_value")
        result = summary.generate(
            iterations=5,
            benchmarks=["math500"],
            overall_improvement=0.10,
            final_accuracy=0.70,
        )
        assert result["custom_key"] == "custom_value"

    def test_benchmark_summary(self):
        summary = BenchmarkSummary()
        data = {
            "math500": {
                "final_accuracy": 0.75,
                "improvement": 0.15,
                "num_tasks": 32,
                "categories": ["algebra"],
            }
        }
        result = summary.generate(data)
        assert "math500" in result["benchmarks"]

    def test_benchmark_summary_markdown(self):
        summary = BenchmarkSummary()
        data = {
            "math500": {
                "final_accuracy": 0.75,
                "improvement": 0.15,
                "num_tasks": 32,
                "categories": ["algebra"],
            }
        }
        md = summary.to_markdown(data)
        assert "Benchmark Summary" in md
        assert "math500" in md

    def test_ablation_summary(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=3,
        )
        result = study.run(sample_benchmarks)

        summary = AblationSummary()
        data = summary.generate(result)
        assert "conditions" in data
        assert "paradigm_ranking" in data

    def test_ablation_summary_markdown(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=3,
        )
        result = study.run(sample_benchmarks)

        summary = AblationSummary()
        md = summary.to_markdown(result)
        assert "Ablation Study Summary" in md

    def test_final_report(self, sample_benchmarks):
        report = FinalReport()
        benchmark_results = {
            bm: {
                "final_accuracy": 0.75,
                "improvement": 0.15,
                "num_tasks": len(sample_benchmarks[bm].tasks),
                "categories": sample_benchmarks[bm].categories,
            }
            for bm in sample_benchmarks
        }

        data = report.generate(
            iterations=3,
            benchmarks=list(sample_benchmarks.keys()),
            overall_improvement=0.15,
            final_accuracy=0.75,
            benchmark_results=benchmark_results,
            collapse_prevention_score=0.85,
            sustainability_score=0.90,
        )
        assert "pipeline" in data
        assert "benchmarks" in data

    def test_final_report_with_ablation(self, sample_benchmarks):
        study = ParadigmAblationStudy(
            agent_factory=lambda cond: MockAgent(cond),
            num_iterations=3,
        )
        ablation_result = study.run(sample_benchmarks)

        report = FinalReport()
        benchmark_results = {
            bm: {"final_accuracy": 0.75, "improvement": 0.15, "num_tasks": 10, "categories": []}
            for bm in sample_benchmarks
        }
        md = report.to_markdown(
            iterations=3,
            benchmarks=list(sample_benchmarks.keys()),
            overall_improvement=0.15,
            final_accuracy=0.75,
            benchmark_results=benchmark_results,
            ablation_result=ablation_result,
        )
        assert "RSI Benchmark Final Report" in md
        assert "Ablation" in md
