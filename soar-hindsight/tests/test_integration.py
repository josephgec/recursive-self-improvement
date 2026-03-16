"""Integration tests covering the full SOAR pipeline and analysis modules."""

import json
import os
import tempfile

import pytest

from src.collection.collector import TrajectoryCollector
from src.collection.database import TrajectoryDatabase
from src.collection.indexer import TrajectoryIndexer
from src.collection.statistics import CorpusStatistics
from src.synthesis.synthesizer import Synthesizer, TrainingPair
from src.synthesis.strategies.direct_solution import DirectSolutionStrategy
from src.synthesis.strategies.error_correction import ErrorCorrectionStrategy
from src.synthesis.strategies.improvement_chain import ImprovementChainStrategy
from src.synthesis.strategies.hindsight_relabel import HindsightRelabelStrategy
from src.synthesis.strategies.crossover_pairs import CrossoverPairsStrategy
from src.synthesis.strategies.pattern_description import PatternDescriptionStrategy
from src.synthesis.quality_filter import QualityFilter
from src.synthesis.deduplicator import Deduplicator
from src.synthesis.formatter import Formatter
from src.finetuning.trainer import Trainer
from src.finetuning.data_loader import DataLoader
from src.finetuning.evaluation import Evaluator
from src.finetuning.model_registry import ModelRegistry
from src.iteration.loop import SOARLoop
from src.iteration.convergence import ConvergenceDetector
from src.iteration.improvement_tracker import ImprovementTracker
from src.analysis.data_quality import DataQualityAnalyzer
from src.analysis.transfer_analysis import TransferAnalyzer
from src.analysis.iteration_dynamics import IterationDynamicsAnalyzer
from src.analysis.report import ReportGenerator
from src.utils.tokenization import count_tokens, truncate_to_tokens, count_tokens_batch
from src.utils.sampling import stratified_sample, reservoir_sample


class TestFullPipeline:
    """Test the complete pipeline from collection through training."""

    def test_collect_synthesize_train(self, fixtures_dir):
        """Full pipeline: collect -> synthesize -> filter -> format -> train."""
        # Step 1: Collect
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        trajectories = collector.collect_from_directory()
        assert len(trajectories) >= 1

        # Step 2: Synthesize
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(min_fitness=0.3), 1.0)
        synthesizer.register_strategy(ErrorCorrectionStrategy(min_attempts=1), 0.8)
        synthesizer.register_strategy(ImprovementChainStrategy(min_steps=1), 0.7)
        pairs = synthesizer.synthesize(trajectories)
        assert len(pairs) >= 1

        # Step 3: Filter
        qf = QualityFilter(
            min_prompt_tokens=1,
            min_completion_tokens=1,
            min_quality_score=0.01,
        )
        filtered = qf.filter(pairs)
        assert len(filtered) >= 1

        # Step 4: Deduplicate
        dedup = Deduplicator(similarity_threshold=0.95)
        unique = dedup.deduplicate(filtered)
        assert len(unique) >= 1

        # Step 5: Format
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.jsonl")
            formatter = Formatter(format_type="openai_jsonl")
            n = formatter.write_jsonl(unique, filepath)
            assert n >= 1

            # Read back
            items = formatter.read_jsonl(filepath)
            assert len(items) == n

        # Step 6: Train
        trainer = Trainer(backend="openai")
        result = trainer.train(unique)
        assert result["status"] == "succeeded"

    def test_collect_and_index(self, fixtures_dir):
        """Test collection and indexing pipeline."""
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        trajectories = collector.collect_from_directory()

        indexer = TrajectoryIndexer()
        indexer.index_many(trajectories)
        assert indexer.total_indexed() == len(trajectories)

        # Query by various criteria
        solved_ids = indexer.lookup_solved(True)
        assert len(solved_ids) >= 1

        stats = CorpusStatistics(trajectories)
        report = stats.full_report()
        assert report["total_trajectories"] == len(trajectories)

    def test_database_roundtrip(self, all_trajectories):
        """Test storing and loading trajectories from database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)

            for traj in all_trajectories:
                db.store(traj)

            assert db.count() == len(all_trajectories)

            # Reload in a new instance
            db2 = TrajectoryDatabase(db_dir=tmpdir)
            db2.initialize()
            assert db2.count() == len(all_trajectories)

            for traj in all_trajectories:
                loaded = db2.load(traj.trajectory_id)
                assert loaded is not None
                assert loaded.best_fitness == traj.best_fitness

    def test_soar_loop_end_to_end(self):
        """Run the full SOAR loop for multiple iterations."""
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(min_fitness=0.1), 1.0)
        synthesizer.register_strategy(ErrorCorrectionStrategy(min_attempts=1), 0.8)
        synthesizer.register_strategy(ImprovementChainStrategy(min_steps=1), 0.7)
        synthesizer.register_strategy(HindsightRelabelStrategy(fitness_threshold=0.1), 0.5)

        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        loop = SOARLoop(
            synthesizer=synthesizer,
            quality_filter=qf,
            max_iterations=3,
            seed=42,
        )
        history = loop.run()
        assert len(history) >= 1

        # Verify we have metrics for each iteration
        for entry in history:
            assert "solve_rate" in entry
            assert "n_pairs_synthesized" in entry

        # Verify tracker
        assert loop.tracker.latest_value is not None

    def test_all_strategies_produce_output(self, all_trajectories, trajectory_with_crossover):
        """Verify every strategy can produce training pairs."""
        all_trajs = all_trajectories + [trajectory_with_crossover]

        strategies = [
            ("direct_solution", DirectSolutionStrategy(min_fitness=0.3)),
            ("error_correction", ErrorCorrectionStrategy(min_attempts=1)),
            ("improvement_chain", ImprovementChainStrategy(min_steps=1)),
            ("hindsight_relabel", HindsightRelabelStrategy(fitness_threshold=0.1)),
            ("crossover_pairs", CrossoverPairsStrategy()),
            ("pattern_description", PatternDescriptionStrategy()),
        ]

        for name, strategy in strategies:
            pairs = strategy.generate(all_trajs)
            assert len(pairs) >= 1, f"Strategy {name} produced no pairs"
            for p in pairs:
                assert p.strategy == name
                assert p.prompt
                assert p.completion


class TestAnalysisModules:
    """Test analysis and reporting modules."""

    def test_data_quality_analyzer(self, sample_training_pairs):
        analyzer = DataQualityAnalyzer(sample_training_pairs)
        assert analyzer.count == 4

        report = analyzer.full_report()
        assert report["total_pairs"] == 4
        assert "token_stats" in report
        assert "strategy_breakdown" in report
        assert "completeness" in report

    def test_data_quality_distribution(self, sample_training_pairs):
        analyzer = DataQualityAnalyzer(sample_training_pairs)
        dist = analyzer.quality_distribution()
        assert isinstance(dist, dict)

    def test_data_quality_completeness(self, sample_training_pairs):
        analyzer = DataQualityAnalyzer(sample_training_pairs)
        checks = analyzer.completeness_check()
        assert checks["empty_prompt"] == 0
        assert checks["empty_completion"] == 0

    def test_data_quality_empty(self):
        analyzer = DataQualityAnalyzer()
        assert analyzer.count == 0
        ts = analyzer.token_stats()
        assert ts["prompt_mean"] == 0

    def test_data_quality_load(self, sample_training_pairs):
        analyzer = DataQualityAnalyzer()
        analyzer.load(sample_training_pairs)
        assert analyzer.count == len(sample_training_pairs)

    def test_transfer_analyzer(self, sample_training_pairs):
        ta = TransferAnalyzer()
        ta.load_pairs(sample_training_pairs)
        coverage = ta.task_coverage()
        assert len(coverage) >= 1

        summary = ta.summary()
        assert summary["total_pairs"] == len(sample_training_pairs)

    def test_transfer_matrix(self, sample_training_pairs):
        ta = TransferAnalyzer()
        ta.load_pairs(sample_training_pairs)
        ta.load_eval_results({
            "task-1": {"zero_shot_solve_rate": 0.5},
            "task-2": {"zero_shot_solve_rate": 0.3},
        })
        matrix = ta.transfer_matrix()
        assert "task-1" in matrix
        assert "task-2" in matrix

    def test_transfer_empty_matrix(self):
        ta = TransferAnalyzer()
        assert ta.transfer_matrix() == {}

    def test_strategy_task_matrix(self, sample_training_pairs):
        ta = TransferAnalyzer()
        ta.load_pairs(sample_training_pairs)
        matrix = ta.strategy_task_matrix()
        assert "direct_solution" in matrix

    def test_quality_by_task(self, sample_training_pairs):
        ta = TransferAnalyzer()
        ta.load_pairs(sample_training_pairs)
        qbt = ta.quality_by_task()
        assert len(qbt) >= 1

    def test_iteration_dynamics(self):
        history = [
            {"iteration": 1, "solve_rate": 0.1, "n_pairs_after_filter": 10,
             "training": {"metrics": {"train_loss_final": 2.0}}, "converged": False},
            {"iteration": 2, "solve_rate": 0.2, "n_pairs_after_filter": 15,
             "training": {"metrics": {"train_loss_final": 1.5}}, "converged": False},
            {"iteration": 3, "solve_rate": 0.25, "n_pairs_after_filter": 20,
             "training": {"metrics": {"train_loss_final": 1.2}}, "converged": False},
        ]
        ida = IterationDynamicsAnalyzer()
        ida.load_history(history)
        assert ida.n_iterations == 3

        sr = ida.solve_rate_trajectory()
        assert len(sr) == 3
        assert sr[0] == (1, 0.1)

        dv = ida.data_volume_trajectory()
        assert len(dv) == 3

        tl = ida.training_loss_trajectory()
        assert len(tl) == 3

        ir = ida.improvement_rate()
        assert len(ir) == 2

        cp = ida.cumulative_pairs()
        assert cp[-1][1] == 45

    def test_iteration_dynamics_convergence(self):
        history = [
            {"iteration": 1, "solve_rate": 0.1, "n_pairs_after_filter": 10, "converged": False},
            {"iteration": 2, "solve_rate": 0.1, "n_pairs_after_filter": 10, "converged": True},
        ]
        ida = IterationDynamicsAnalyzer()
        ida.load_history(history)
        conv = ida.convergence_analysis()
        assert conv["converged"] is True

    def test_iteration_dynamics_empty(self):
        ida = IterationDynamicsAnalyzer()
        assert ida.n_iterations == 0
        assert ida.solve_rate_trajectory() == []
        conv = ida.convergence_analysis()
        assert conv["converged"] is False

    def test_iteration_dynamics_full_report(self):
        history = [{"iteration": 1, "solve_rate": 0.1, "n_pairs_after_filter": 10, "converged": False}]
        ida = IterationDynamicsAnalyzer()
        ida.load_history(history)
        report = ida.full_report()
        assert "n_iterations" in report
        assert "convergence" in report

    def test_report_generator(self, sample_training_pairs):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            history = [
                {"iteration": 1, "solve_rate": 0.2, "n_pairs_after_filter": 10, "converged": False},
            ]
            report = gen.generate(
                pairs=sample_training_pairs,
                iteration_history=history,
            )
            assert "sections" in report
            assert "summary" in report

            # Save
            filepath = gen.save(report)
            assert os.path.exists(filepath)

            # Text format
            text = gen.format_text(report)
            assert "SOAR Hindsight Report" in text

    def test_report_generator_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            report = gen.generate()
            assert report["sections"] == {}

    def test_report_generator_with_eval_results(self, sample_training_pairs):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            eval_results = {"task-1": {"zero_shot_solve_rate": 0.5}}
            report = gen.generate(
                pairs=sample_training_pairs,
                eval_results=eval_results,
            )
            assert "transfer" in report["sections"]


class TestUtilities:
    """Test utility modules."""

    def test_count_tokens(self):
        assert count_tokens("") == 0
        assert count_tokens("hello world") > 0
        assert count_tokens("one two three four five") >= 5

    def test_truncate_to_tokens(self):
        text = "word " * 100
        truncated = truncate_to_tokens(text, 10)
        assert len(truncated) < len(text)
        assert truncated.endswith("...")

    def test_truncate_short_text(self):
        text = "hello"
        result = truncate_to_tokens(text, 100)
        assert result == text

    def test_truncate_empty(self):
        assert truncate_to_tokens("", 10) == ""

    def test_count_tokens_batch(self):
        texts = ["hello world", "foo bar baz", ""]
        counts = count_tokens_batch(texts)
        assert len(counts) == 3
        assert counts[0] > 0
        assert counts[2] == 0

    def test_stratified_sample(self, sample_training_pairs):
        result = stratified_sample(
            sample_training_pairs,
            key_fn=lambda p: p.strategy,
            n=3,
        )
        assert len(result) == 3
        strategies = set(p.strategy for p in result)
        assert len(strategies) >= 1

    def test_stratified_sample_n_larger(self, sample_training_pairs):
        result = stratified_sample(
            sample_training_pairs,
            key_fn=lambda p: p.strategy,
            n=100,
        )
        assert len(result) == len(sample_training_pairs)

    def test_reservoir_sample(self):
        items = list(range(100))
        result = reservoir_sample(items, 10)
        assert len(result) == 10
        assert all(isinstance(x, int) for x in result)

    def test_reservoir_sample_n_larger(self):
        items = [1, 2, 3]
        result = reservoir_sample(items, 10)
        assert len(result) == 3

    def test_stratified_sample_deterministic(self, sample_training_pairs):
        r1 = stratified_sample(sample_training_pairs, lambda p: p.strategy, 3, seed=42)
        r2 = stratified_sample(sample_training_pairs, lambda p: p.strategy, 3, seed=42)
        assert [p.pair_id for p in r1] == [p.pair_id for p in r2]

    def test_reservoir_deterministic(self):
        items = list(range(50))
        r1 = reservoir_sample(items, 5, seed=42)
        r2 = reservoir_sample(items, 5, seed=42)
        assert r1 == r2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_trajectory(self):
        from src.collection.trajectory import SearchTrajectory
        traj = SearchTrajectory()
        assert traj.best_individual is None
        assert traj.solved_individuals == []
        assert traj.failed_individuals == []
        assert traj.generations == []
        chain = traj.extract_improvement_chain()
        assert chain == []

    def test_single_individual_trajectory(self):
        from src.collection.trajectory import SearchTrajectory, IndividualRecord, TaskSpec
        traj = SearchTrajectory(
            task=TaskSpec(task_id="t", description="d"),
            individuals=[IndividualRecord(code="x", fitness=0.5)],
            best_fitness=0.5,
        )
        chain = traj.extract_improvement_chain()
        assert chain == []

    def test_formatter_write_to_current_dir(self, sample_training_pairs):
        formatter = Formatter()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "out.jsonl")
            n = formatter.write_jsonl(sample_training_pairs, filepath)
            assert n == len(sample_training_pairs)

    def test_training_pair_from_dict_defaults(self):
        pair = TrainingPair.from_dict({})
        assert pair.strategy == ""
        assert pair.quality_score == 0.0

    def test_convergence_never_converges_with_always_improving(self):
        cd = ConvergenceDetector(patience=3, min_improvement=0.01)
        for i in range(10):
            cd.check(i * 0.1)
        assert cd.is_converged is False

    def test_soar_loop_zero_iterations(self):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
        loop = SOARLoop(synthesizer=synthesizer, max_iterations=0)
        history = loop.run()
        assert len(history) == 0

    def test_multiple_collect_calls(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        t1 = collector.collect_from_directory()
        t2 = collector.collect_from_directory()
        assert collector.count == len(t1) + len(t2)

    def test_indexer_fitness_bucket_edge_cases(self):
        from src.collection.indexer import TrajectoryIndexer
        assert TrajectoryIndexer._fitness_bucket(-0.5) == "0.0-0.1"
        assert TrajectoryIndexer._fitness_bucket(1.5) == "1.0-1.0"
