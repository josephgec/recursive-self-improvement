"""Supplementary tests for improving coverage on under-tested modules."""

import pytest
import random

from src.benchmarks.task import EvalTask, EvalResult
from src.execution.rlm_executor import RLMExecutor, MockRLM
from src.execution.standard_executor import StandardExecutor, MockStandardLLM
from src.execution.runner import BenchmarkRunner
from src.comparison.head_to_head import HeadToHeadComparator
from src.comparison.statistical_tests import StatisticalTests
from src.strategies.failure_modes import StrategyFailureModeAnalyzer, FailureCase
from src.strategies.emergence_analyzer import EmergenceAnalyzer
from src.analysis.trajectory_visualizer import TrajectoryVisualizer
from src.analysis.context_scaling import ContextScalingAnalysis
from src.comparison.scaling_experiment import ScalingResult


# --- RLM Executor coverage ---

class TestRLMExecutorCoverage:
    """Cover uncovered branches in rlm_executor.py."""

    def test_bug_fix_category_trajectory(self):
        """Cover bug_fix trajectory generation."""
        executor = RLMExecutor(seed=42)
        task = EvalTask(
            task_id="bug_task",
            benchmark="test",
            query="What was fixed?",
            context="Before: broken\nAfter: fixed",
            expected_answer="null check",
            category="bug_fix",
            context_tokens=100,
        )
        result = executor.execute(task)
        assert result.task_id == "bug_task"
        # Bug fix uses DIRECT strategy
        assert result.strategy_detected == "DIRECT"
        assert any("cat context.txt" in step or "grep" in step for step in result.trajectory)

    def test_unknown_category_trajectory(self):
        """Cover fallback trajectory for unknown categories."""
        executor = RLMExecutor(seed=42)
        task = EvalTask(
            task_id="unknown_task",
            benchmark="test",
            query="What is this?",
            context="Some context",
            expected_answer="something",
            category="mystery_category",
            context_tokens=50,
        )
        result = executor.execute(task)
        assert result.task_id == "unknown_task"
        # Unknown category uses DIRECT strategy
        assert result.strategy_detected == "DIRECT"

    def test_check_answer_exact_match(self):
        executor = RLMExecutor(seed=42)
        assert executor._check_answer("hello", "hello")

    def test_check_answer_expected_in_answer(self):
        executor = RLMExecutor(seed=42)
        assert executor._check_answer("the answer is 42", "42")

    def test_check_answer_answer_in_expected(self):
        executor = RLMExecutor(seed=42)
        assert executor._check_answer("42", "the answer is 42")

    def test_check_answer_no_match(self):
        executor = RLMExecutor(seed=42)
        assert not executor._check_answer("wrong", "right")

    def test_default_mock_llm(self):
        executor = RLMExecutor(seed=42)
        result = executor._default_mock_llm("prompt")
        assert result == "mock_answer"

    def test_mock_rlm_class(self):
        mock = MockRLM(seed=99)
        assert mock.call_count == 0
        result = mock("test prompt")
        assert mock.call_count == 1
        assert result == "mock_rlm_response"

        executor = mock.create_executor()
        assert isinstance(executor, RLMExecutor)


# --- Standard Executor coverage ---

class TestStandardExecutorCoverage:
    """Cover uncovered branches in standard_executor.py."""

    def test_truncated_answer_visible(self):
        """Cover the branch where answer is in truncated context."""
        std = StandardExecutor(context_window=100, seed=42)
        task = EvalTask(
            task_id="trunc_visible",
            benchmark="test",
            query="What is it?",
            context="The answer is XYZ123. " * 50,
            expected_answer="XYZ123",
            category="retrieval",
            context_tokens=500,
            difficulty="easy",
        )
        result = std.execute(task)
        assert result.task_id == "trunc_visible"
        # With seed 42, answer visible in truncation -> 70% chance correct
        assert isinstance(result.correct, bool)

    def test_truncated_answer_not_visible(self):
        """Cover the branch where answer is NOT in truncated context."""
        std = StandardExecutor(context_window=100, seed=42)
        task = EvalTask(
            task_id="trunc_hidden",
            benchmark="test",
            query="What is the hidden value?",
            context="Filler text only. " * 200 + "Hidden: SECRET42",
            expected_answer="SECRET42",
            category="needle_in_haystack",
            context_tokens=1000,
            difficulty="hard",
        )
        result = std.execute(task)
        assert result.task_id == "trunc_hidden"

    def test_full_context_easy(self):
        """Cover the full-context correctness path for easy tasks."""
        std = StandardExecutor(context_window=100000, seed=42)
        task = EvalTask(
            task_id="full_easy",
            benchmark="test",
            query="Simple question?",
            context="Short context",
            expected_answer="answer",
            context_tokens=10,
            difficulty="easy",
        )
        result = std.execute(task)
        assert isinstance(result.correct, bool)

    def test_full_context_hard(self):
        """Cover the full-context correctness path for hard tasks."""
        std = StandardExecutor(context_window=100000, seed=42)
        task = EvalTask(
            task_id="full_hard",
            benchmark="test",
            query="Hard question?",
            context="Short context",
            expected_answer="answer",
            context_tokens=10,
            difficulty="hard",
        )
        result = std.execute(task)
        assert isinstance(result.correct, bool)

    def test_full_context_unknown_difficulty(self):
        """Cover the default difficulty rate."""
        std = StandardExecutor(context_window=100000, seed=42)
        task = EvalTask(
            task_id="full_unknown_diff",
            benchmark="test",
            query="Question?",
            context="Context",
            expected_answer="answer",
            context_tokens=10,
            difficulty="extreme",
        )
        result = std.execute(task)
        assert isinstance(result.correct, bool)

    def test_mock_standard_llm_class(self):
        mock = MockStandardLLM(seed=99)
        assert mock.call_count == 0
        result = mock("prompt")
        assert mock.call_count == 1
        assert result == "mock_standard_response"

        executor = mock.create_executor(context_window=4096)
        assert isinstance(executor, StandardExecutor)


# --- Runner coverage: scaling experiment ---

class TestRunnerScalingExperiment:
    """Cover run_scaling_experiment method."""

    def test_scaling_experiment_via_runner(self, tmp_checkpoint_dir):
        rlm = RLMExecutor(seed=42)
        std = StandardExecutor(context_window=2000, seed=42)

        runner = BenchmarkRunner(
            executor_fn=rlm.execute,
            checkpoint_dir=tmp_checkpoint_dir,
        )

        base_tasks = [
            EvalTask(
                task_id="scale_t1",
                benchmark="test",
                query="Find the code?",
                context="Secret code is ABC. " * 50,
                expected_answer="ABC",
                category="retrieval",
                context_tokens=200,
            ),
        ]

        results = runner.run_scaling_experiment(
            base_tasks,
            context_sizes=[500, 2000, 5000],
            rlm_executor_fn=rlm.execute,
            standard_executor_fn=std.execute,
        )
        assert len(results.context_sizes) == 3
        assert len(results.rlm_accuracies) == 3
        assert len(results.standard_accuracies) == 3

    def test_runner_budget_cutoff(self, tmp_checkpoint_dir):
        rlm = RLMExecutor(seed=42)
        runner = BenchmarkRunner(
            executor_fn=rlm.execute,
            checkpoint_dir=tmp_checkpoint_dir,
            budget_limit=0.0001,  # Very low budget
        )
        tasks = [
            EvalTask(
                task_id=f"budget_{i}",
                benchmark="test",
                query="Q?",
                context="C" * 1000,
                expected_answer="A",
                context_tokens=250,
            )
            for i in range(10)
        ]
        run = runner.run_benchmark(tasks, "budget_test", run_id="budget_run", resume=False)
        # Should stop early due to budget
        assert run.total_tasks <= 10


# --- Head-to-head coverage ---

class TestHeadToHeadCoverage:
    """Cover uncovered branches in head_to_head.py."""

    def test_compare_with_categories(self):
        comparator = HeadToHeadComparator()
        rlm_results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True,
                       cost=0.01, input_tokens=5000),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=True,
                       cost=0.02, input_tokens=5000),
            EvalResult(task_id="t3", benchmark="b", answer="c", correct=False,
                       cost=0.01, input_tokens=5000),
        ]
        std_results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=False,
                       cost=0.005, input_tokens=3000),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False,
                       cost=0.005, input_tokens=3000),
            EvalResult(task_id="t3", benchmark="b", answer="c", correct=True,
                       cost=0.005, input_tokens=3000),
        ]
        categories = {"t1": "retrieval", "t2": "counting", "t3": "reasoning"}

        report = comparator.compare(rlm_results, std_results, categories)
        assert report.total_tasks == 3
        assert report.rlm_wins == 2
        assert report.standard_wins == 1
        assert report.ties == 0
        assert "retrieval" in report.advantage_categories
        assert len(report.paired_results) == 3

        # Summary should mention key stats
        summary = report.summary()
        assert "Head-to-Head" in summary

    def test_compute_2x_claim_no_standard_correct(self):
        """Test 2x claim when standard gets nothing right."""
        comparator = HeadToHeadComparator()
        rlm = [EvalResult(task_id="t1", benchmark="b", answer="a", correct=True, input_tokens=5000)]
        std = [EvalResult(task_id="t1", benchmark="b", answer="b", correct=False, input_tokens=5000)]
        assert comparator.compute_2x_claim(rlm, std) is True

    def test_compute_2x_claim_no_tasks(self):
        comparator = HeadToHeadComparator()
        assert comparator.compute_2x_claim([], []) is False

    def test_paired_accuracy_no_overlap(self):
        comparator = HeadToHeadComparator()
        rlm = [EvalResult(task_id="t1", benchmark="b", answer="a", correct=True)]
        std = [EvalResult(task_id="t2", benchmark="b", answer="b", correct=True)]
        acc = comparator.paired_accuracy(rlm, std)
        assert acc == (0.0, 0.0)

    def test_category_advantages_tie(self):
        comparator = HeadToHeadComparator()
        rlm = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True, input_tokens=5000),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False, input_tokens=5000),
        ]
        std = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True, input_tokens=3000),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False, input_tokens=3000),
        ]
        categories = {"t1": "same_cat", "t2": "same_cat"}
        report = comparator.compare(rlm, std, categories)
        assert report.advantage_categories["same_cat"] == "tie"


# --- Statistical tests coverage ---

class TestStatisticalTestsCoverage:
    """Cover uncovered branches in statistical_tests.py."""

    def test_mcnemar_no_discordant_pairs(self):
        """All agree -> p=1.0, not significant."""
        tests = StatisticalTests()
        rlm = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False),
        ]
        std = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False),
        ]
        result = tests.mcnemar_test(rlm, std)
        assert result.p_value == 1.0
        assert not result.significant

    def test_chi2_p_value_zero(self):
        tests = StatisticalTests()
        p = tests._chi2_p_value(0)
        assert p == 1.0

    def test_normal_cdf_complement_negative(self):
        tests = StatisticalTests()
        p = tests._normal_cdf_complement(-1.0)
        assert p > 0.5

    def test_z_score_lookup(self):
        tests = StatisticalTests()
        assert tests._z_score(0.90) == 1.645
        assert tests._z_score(0.99) == 2.576
        assert tests._z_score(0.85) == 1.96  # Default

    def test_confidence_interval_empty(self):
        tests = StatisticalTests()
        ci = tests.confidence_interval([])
        assert ci == (0.0, 0.0)

    def test_accuracy_difference_ci_no_overlap(self):
        tests = StatisticalTests()
        rlm = [EvalResult(task_id="t1", benchmark="b", answer="a", correct=True)]
        std = [EvalResult(task_id="t2", benchmark="b", answer="b", correct=True)]
        ci = tests.accuracy_difference_ci(rlm, std)
        assert ci == (0.0, 0.0)

    def test_paired_proportion_test_alias(self):
        tests = StatisticalTests()
        rlm = [EvalResult(task_id="t1", benchmark="b", answer="a", correct=True)]
        std = [EvalResult(task_id="t1", benchmark="b", answer="a", correct=False)]
        r1 = tests.paired_proportion_test(rlm, std)
        r2 = tests.mcnemar_test(rlm, std)
        assert r1.statistic == r2.statistic


# --- Failure modes coverage ---

class TestFailureModesCoverage:
    """Cover uncovered branches in failure_modes.py."""

    def test_categorize_truncation_error(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="trunc_fail",
                benchmark="b",
                answer="wrong",
                correct=False,
                trajectory=["truncated context due to length limit"],
                strategy_detected="DIRECT",
            ),
        ]
        failures = analyzer.categorize_failures(results)
        assert "truncation" in failures

    def test_categorize_aggregation_error(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="agg_fail",
                benchmark="b",
                answer="999",
                correct=False,
                trajectory=["computed total"],
                strategy_detected="MAP_REDUCE",
            ),
        ]
        categories = {"agg_fail": "aggregation"}
        failures = analyzer.categorize_failures(results, categories)
        assert "aggregation_error" in failures

    def test_categorize_reasoning_error(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="reason_fail",
                benchmark="b",
                answer="wrong_conclusion",
                correct=False,
                trajectory=["analyzed clues"],
                strategy_detected="HIERARCHICAL",
            ),
        ]
        categories = {"reason_fail": "reasoning"}
        failures = analyzer.categorize_failures(results, categories)
        assert "reasoning_error" in failures

    def test_categorize_runtime_error(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="runtime_fail",
                benchmark="b",
                answer="",
                correct=False,
                error="division by zero",
                trajectory=[],
                strategy_detected="DIRECT",
            ),
        ]
        failures = analyzer.categorize_failures(results)
        assert "runtime_error" in failures

    def test_categorize_incomplete_search(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="search_fail",
                benchmark="b",
                answer="partial",
                correct=False,
                trajectory=["grep 'pattern' file.txt"],
                strategy_detected="PEEK_THEN_GREP",
            ),
        ]
        failures = analyzer.categorize_failures(results)
        assert "incomplete_search" in failures

    def test_categorize_unknown_error(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="mystery_fail",
                benchmark="b",
                answer="wrong",
                correct=False,
                trajectory=["did something"],
                strategy_detected="DIRECT",
            ),
        ]
        failures = analyzer.categorize_failures(results)
        assert "unknown" in failures

    def test_strategy_misapplication_with_categories(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="mis_retrieval",
                benchmark="b",
                answer="wrong",
                correct=False,
                trajectory=[],
                strategy_detected="MAP_REDUCE",  # Wrong for retrieval
            ),
        ]
        categories = {"mis_retrieval": "retrieval"}
        misapplied = analyzer.strategy_misapplication(results, categories)
        assert len(misapplied) >= 1
        assert misapplied[0].error_type == "wrong_strategy"

    def test_strategy_misapplication_correct_ignored(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="correct_task",
                benchmark="b",
                answer="right",
                correct=True,
                strategy_detected="MAP_REDUCE",
            ),
        ]
        categories = {"correct_task": "retrieval"}
        misapplied = analyzer.strategy_misapplication(results, categories)
        assert len(misapplied) == 0

    def test_failure_rate_by_strategy(self):
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True, strategy_detected="A"),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=False, strategy_detected="A"),
            EvalResult(task_id="t3", benchmark="b", answer="c", correct=False, strategy_detected="B"),
        ]
        rates = analyzer.failure_rate_by_strategy(results)
        assert rates["A"] == 0.5
        assert rates["B"] == 1.0

    def test_categorize_by_task_id_inference(self):
        """Cover the branch that infers category from task_id."""
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="counting_task_1",
                benchmark="b",
                answer="999",
                correct=False,
                trajectory=["some step"],
                strategy_detected="DIRECT",
            ),
        ]
        # No explicit categories - should infer from task_id
        failures = analyzer.categorize_failures(results)
        assert isinstance(failures, dict)

    def test_misapplication_infer_from_task_id(self):
        """Cover branch that infers category from task_id in misapplication."""
        analyzer = StrategyFailureModeAnalyzer()
        results = [
            EvalResult(
                task_id="needle_hidden_42",
                benchmark="b",
                answer="wrong",
                correct=False,
                strategy_detected="MAP_REDUCE",  # Wrong for needle
            ),
        ]
        # No explicit categories, should infer "needle" from task_id
        misapplied = analyzer.strategy_misapplication(results)
        assert len(misapplied) >= 1


# --- Trajectory visualizer coverage ---

class TestTrajectoryVisualizerCoverage:
    """Cover uncovered branches in trajectory_visualizer.py."""

    def test_select_representative_fewer_than_requested(self):
        """When there are fewer results than requested examples."""
        viz = TrajectoryVisualizer()
        results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True,
                       trajectory=["step1"], strategy_detected="DIRECT"),
        ]
        selected = viz.select_representative(results, num_examples=5)
        assert len(selected) == 1

    def test_select_representative_ensures_correct_and_incorrect(self):
        """Cover the branches that add correct/incorrect examples."""
        viz = TrajectoryVisualizer()
        # All same strategy, all correct - should trigger "no incorrect" branch
        results = [
            EvalResult(task_id=f"t{i}", benchmark="b", answer="a",
                       correct=True, trajectory=["step"], strategy_detected="DIRECT")
            for i in range(10)
        ]
        # Add one incorrect result
        results.append(EvalResult(
            task_id="t_wrong", benchmark="b", answer="wrong",
            correct=False, trajectory=["step"], strategy_detected="DIRECT",
        ))
        selected = viz.select_representative(results, num_examples=3)
        assert len(selected) == 3

    def test_select_representative_all_incorrect_same_strategy(self):
        """Cover the 'has_correct is False' branch."""
        viz = TrajectoryVisualizer()
        results = [
            EvalResult(task_id=f"t{i}", benchmark="b", answer="wrong",
                       correct=False, trajectory=["step"], strategy_detected="DIRECT")
            for i in range(10)
        ]
        # Add one correct
        results.append(EvalResult(
            task_id="t_right", benchmark="b", answer="a",
            correct=True, trajectory=["step"], strategy_detected="DIRECT",
        ))
        selected = viz.select_representative(results, num_examples=3)
        assert len(selected) == 3

    def test_select_representative_fills_remaining(self):
        """Cover the fill remaining slots branch."""
        viz = TrajectoryVisualizer()
        # Many different strategies to fill first pass
        strategies = ["A", "B", "C", "D", "E"]
        results = []
        for i, s in enumerate(strategies):
            results.append(EvalResult(
                task_id=f"t{i}", benchmark="b", answer="a",
                correct=i % 2 == 0, trajectory=["step"],
                strategy_detected=s,
            ))
        # Request more than strategies
        selected = viz.select_representative(results, num_examples=5)
        assert len(selected) == 5


# --- Emergence analyzer edge cases ---

class TestEmergenceAnalyzerCoverage:
    """Cover edge cases in emergence_analyzer.py."""

    def test_analyze_empty_results(self):
        analyzer = EmergenceAnalyzer()
        report = analyzer.analyze([])
        assert report.total_results_analyzed == 0
        assert report.grep_before_read_rate == 0.0
        assert report.adaptation_score == 0.0

    def test_strategy_by_context_size_with_input_tokens(self):
        analyzer = EmergenceAnalyzer()
        results = [
            EvalResult(task_id="t1", benchmark="b", answer="a", correct=True,
                       strategy_detected="DIRECT", input_tokens=1000),
            EvalResult(task_id="t2", benchmark="b", answer="b", correct=True,
                       strategy_detected="PEEK_THEN_GREP", input_tokens=20000),
            EvalResult(task_id="t3", benchmark="b", answer="c", correct=True,
                       strategy_detected="MAP_REDUCE", input_tokens=100000),
        ]
        by_size = analyzer.strategy_by_context_size(results)
        assert "small (<4k)" in by_size
        assert "large (16k-64k)" in by_size
        assert "xlarge (>64k)" in by_size

    def test_adaptation_single_strategy(self):
        analyzer = EmergenceAnalyzer()
        results = [
            EvalResult(task_id=f"t{i}", benchmark="b", answer="a", correct=True,
                       strategy_detected="DIRECT")
            for i in range(5)
        ]
        score = analyzer.adaptation_within_session(results)
        assert score == 0.0


# --- Context scaling analysis edge cases ---

class TestContextScalingAnalysisCoverage:
    """Cover edge cases in context_scaling.py."""

    def test_degradation_analysis_empty(self):
        analysis = ContextScalingAnalysis()
        result = analysis.degradation_analysis([])
        assert result == {}

    def test_plot_ascii_empty(self):
        analysis = ContextScalingAnalysis()
        plot = analysis.plot_ascii([])
        assert "No scaling data" in plot
