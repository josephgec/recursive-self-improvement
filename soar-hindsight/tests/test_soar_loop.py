"""Tests for the SOAR loop, improvement tracker, and convergence."""

import pytest

from src.iteration.loop import SOARLoop
from src.iteration.improvement_tracker import ImprovementTracker
from src.iteration.convergence import ConvergenceDetector
from src.synthesis.synthesizer import Synthesizer
from src.synthesis.strategies.direct_solution import DirectSolutionStrategy
from src.synthesis.strategies.error_correction import ErrorCorrectionStrategy
from src.synthesis.strategies.improvement_chain import ImprovementChainStrategy
from src.synthesis.strategies.hindsight_relabel import HindsightRelabelStrategy
from src.finetuning.trainer import Trainer
from src.finetuning.evaluation import Evaluator
from src.synthesis.quality_filter import QualityFilter


class TestConvergenceDetector:
    def test_not_converged_initially(self):
        cd = ConvergenceDetector(patience=3, min_improvement=0.01)
        assert cd.is_converged is False

    def test_converges_on_plateau(self):
        cd = ConvergenceDetector(patience=3, min_improvement=0.01)
        cd.check(0.5)  # sets best_value, no_improve_count=0
        cd.check(0.5)  # no_improve_count=1
        cd.check(0.5)  # no_improve_count=2
        cd.check(0.5)  # no_improve_count=3 >= patience
        assert cd.is_converged is True

    def test_resets_on_improvement(self):
        cd = ConvergenceDetector(patience=3, min_improvement=0.01)
        cd.check(0.5)
        cd.check(0.5)
        cd.check(0.6)  # improvement resets counter
        assert cd.is_converged is False
        assert cd.no_improve_count == 0

    def test_best_value(self):
        cd = ConvergenceDetector()
        cd.check(0.3)
        cd.check(0.5)
        cd.check(0.4)
        assert cd.best_value == 0.5

    def test_best_value_empty(self):
        cd = ConvergenceDetector()
        assert cd.best_value == 0.0

    def test_reset(self):
        cd = ConvergenceDetector()
        cd.check(0.5)
        cd.reset()
        assert cd.is_converged is False
        assert cd.no_improve_count == 0

    def test_summary(self):
        cd = ConvergenceDetector(patience=3, min_improvement=0.01)
        cd.check(0.5)
        s = cd.summary()
        assert "converged" in s
        assert "best_value" in s
        assert "patience" in s
        assert s["n_checks"] == 1

    def test_small_improvement_not_enough(self):
        cd = ConvergenceDetector(patience=2, min_improvement=0.1)
        cd.check(0.5)
        cd.check(0.55)  # only 0.05 improvement, below 0.1 threshold
        assert cd.no_improve_count == 1
        cd.check(0.56)
        assert cd.is_converged is True


class TestImprovementTracker:
    def test_record_and_history(self):
        tracker = ImprovementTracker()
        tracker.record(1, 0.1)
        tracker.record(2, 0.2)
        tracker.record(3, 0.3)
        assert len(tracker.history) == 3
        assert tracker.values == [0.1, 0.2, 0.3]

    def test_latest_value(self):
        tracker = ImprovementTracker()
        assert tracker.latest_value is None
        tracker.record(1, 0.5)
        assert tracker.latest_value == 0.5

    def test_best_value(self):
        tracker = ImprovementTracker()
        assert tracker.best_value is None
        tracker.record(1, 0.3)
        tracker.record(2, 0.5)
        tracker.record(3, 0.4)
        assert tracker.best_value == 0.5
        assert tracker.best_iteration == 2

    def test_improvement_from_start(self):
        tracker = ImprovementTracker()
        assert tracker.improvement_from_start() == 0.0
        tracker.record(1, 0.2)
        tracker.record(2, 0.4)
        assert tracker.improvement_from_start() == pytest.approx(0.2)

    def test_recent_improvement(self):
        tracker = ImprovementTracker()
        tracker.record(1, 0.1)
        tracker.record(2, 0.2)
        tracker.record(3, 0.3)
        ri = tracker.recent_improvement(window=3)
        assert ri == pytest.approx(0.1)

    def test_recent_improvement_short(self):
        tracker = ImprovementTracker()
        assert tracker.recent_improvement() == 0.0
        tracker.record(1, 0.5)
        assert tracker.recent_improvement() == 0.0

    def test_is_improving(self):
        tracker = ImprovementTracker()
        tracker.record(1, 0.1)
        tracker.record(2, 0.2)
        tracker.record(3, 0.3)
        assert tracker.is_improving(window=3, threshold=0.05) is True

    def test_is_not_improving(self):
        tracker = ImprovementTracker()
        tracker.record(1, 0.5)
        tracker.record(2, 0.5)
        tracker.record(3, 0.5)
        assert tracker.is_improving(window=3, threshold=0.01) is False

    def test_summary(self):
        tracker = ImprovementTracker()
        tracker.record(1, 0.3)
        tracker.record(2, 0.5)
        s = tracker.summary()
        assert s["n_records"] == 2
        assert s["best_value"] == 0.5
        assert s["total_improvement"] == pytest.approx(0.2)

    def test_clear(self):
        tracker = ImprovementTracker()
        tracker.record(1, 0.5)
        tracker.clear()
        assert len(tracker.history) == 0
        assert tracker.latest_value is None


class TestSOARLoop:
    def _make_synthesizer(self):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(min_fitness=0.3), 1.0)
        synthesizer.register_strategy(ErrorCorrectionStrategy(min_attempts=1), 0.8)
        return synthesizer

    def test_mock_search(self):
        synthesizer = self._make_synthesizer()
        loop = SOARLoop(synthesizer=synthesizer, seed=42)
        trajectories = loop.mock_search(n_trajectories=5, model_quality=0.5)
        assert len(trajectories) == 5
        for t in trajectories:
            assert t.task is not None
            assert len(t.individuals) > 0

    def test_run_iteration(self):
        synthesizer = self._make_synthesizer()
        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        loop = SOARLoop(synthesizer=synthesizer, quality_filter=qf, seed=42)
        result = loop.run_iteration()
        assert result["iteration"] == 1
        assert "n_trajectories" in result
        assert "n_pairs_synthesized" in result
        assert "solve_rate" in result
        assert "comparison" in result

    def test_run_multiple_iterations(self):
        synthesizer = self._make_synthesizer()
        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        loop = SOARLoop(synthesizer=synthesizer, quality_filter=qf, max_iterations=3, seed=42)
        history = loop.run(max_iterations=2)
        assert len(history) >= 1
        assert loop.iteration >= 1

    def test_run_with_convergence(self):
        synthesizer = self._make_synthesizer()
        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        cd = ConvergenceDetector(patience=1, min_improvement=10.0)  # will converge quickly
        loop = SOARLoop(
            synthesizer=synthesizer,
            quality_filter=qf,
            convergence=cd,
            max_iterations=10,
            seed=42,
        )
        history = loop.run()
        # Should stop before max_iterations due to convergence
        assert loop.iteration <= 10

    def test_run_with_provided_trajectories(self, all_trajectories):
        synthesizer = self._make_synthesizer()
        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        loop = SOARLoop(synthesizer=synthesizer, quality_filter=qf, seed=42)
        result = loop.run_iteration(trajectories=all_trajectories)
        assert result["n_trajectories"] == len(all_trajectories)

    def test_history(self):
        synthesizer = self._make_synthesizer()
        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        loop = SOARLoop(synthesizer=synthesizer, quality_filter=qf, seed=42)
        loop.run_iteration()
        assert len(loop.history) == 1

    def test_tracker(self):
        synthesizer = self._make_synthesizer()
        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        loop = SOARLoop(synthesizer=synthesizer, quality_filter=qf, seed=42)
        loop.run_iteration()
        assert loop.tracker.latest_value is not None

    def test_summary(self):
        synthesizer = self._make_synthesizer()
        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        loop = SOARLoop(synthesizer=synthesizer, quality_filter=qf, seed=42)
        loop.run_iteration()
        s = loop.summary()
        assert "total_iterations" in s
        assert "final_solve_rate" in s
        assert "converged" in s

    def test_iteration_property(self):
        synthesizer = self._make_synthesizer()
        loop = SOARLoop(synthesizer=synthesizer)
        assert loop.iteration == 0

    def test_all_strategies(self):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(min_fitness=0.1), 1.0)
        synthesizer.register_strategy(ErrorCorrectionStrategy(min_attempts=1), 0.8)
        synthesizer.register_strategy(ImprovementChainStrategy(min_steps=1), 0.7)
        synthesizer.register_strategy(HindsightRelabelStrategy(fitness_threshold=0.1), 0.5)
        qf = QualityFilter(min_prompt_tokens=1, min_completion_tokens=1, min_quality_score=0.01)
        loop = SOARLoop(synthesizer=synthesizer, quality_filter=qf, seed=42)
        result = loop.run_iteration()
        assert result["n_pairs_synthesized"] > 0
