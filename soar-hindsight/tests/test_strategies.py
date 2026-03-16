"""Tests for all synthesis strategies."""

import pytest

from src.synthesis.strategies.direct_solution import DirectSolutionStrategy
from src.synthesis.strategies.error_correction import ErrorCorrectionStrategy
from src.synthesis.strategies.improvement_chain import ImprovementChainStrategy
from src.synthesis.strategies.hindsight_relabel import HindsightRelabelStrategy
from src.synthesis.strategies.crossover_pairs import CrossoverPairsStrategy
from src.synthesis.strategies.pattern_description import PatternDescriptionStrategy
from src.synthesis.synthesizer import Synthesizer, TrainingPair
from src.collection.trajectory import (
    SearchTrajectory,
    IndividualRecord,
    ImprovementStep,
    TaskSpec,
)


class TestDirectSolutionStrategy:
    def test_generates_from_solved(self, solved_trajectory):
        strategy = DirectSolutionStrategy(min_fitness=0.5)
        pairs = strategy.generate([solved_trajectory])
        assert len(pairs) >= 1
        for p in pairs:
            assert p.strategy == "direct_solution"
            assert p.prompt
            assert p.completion
            assert p.quality_score >= 0.5

    def test_skips_low_fitness(self, failed_trajectory):
        strategy = DirectSolutionStrategy(min_fitness=0.9)
        pairs = strategy.generate([failed_trajectory])
        assert len(pairs) == 0

    def test_skips_no_task(self):
        traj = SearchTrajectory(
            individuals=[IndividualRecord(fitness=1.0, code="code")],
            best_fitness=1.0,
        )
        strategy = DirectSolutionStrategy()
        pairs = strategy.generate([traj])
        assert len(pairs) == 0

    def test_skips_empty_code(self):
        task = TaskSpec(task_id="t", description="desc")
        traj = SearchTrajectory(
            task=task,
            individuals=[IndividualRecord(fitness=1.0, code="")],
            best_fitness=1.0,
        )
        strategy = DirectSolutionStrategy()
        pairs = strategy.generate([traj])
        assert len(pairs) == 0

    def test_token_counts(self, solved_trajectory):
        strategy = DirectSolutionStrategy()
        pairs = strategy.generate([solved_trajectory])
        for p in pairs:
            assert p.prompt_tokens > 0
            assert p.completion_tokens > 0

    def test_name(self):
        assert DirectSolutionStrategy.name == "direct_solution"


class TestErrorCorrectionStrategy:
    def test_generates_from_errors(self, failed_trajectory):
        strategy = ErrorCorrectionStrategy(min_attempts=1)
        pairs = strategy.generate([failed_trajectory])
        assert len(pairs) >= 1
        for p in pairs:
            assert p.strategy == "error_correction"

    def test_generates_from_improvement_steps(self, solved_trajectory):
        # The solved_trajectory has an error individual that was later fixed
        strategy = ErrorCorrectionStrategy(min_attempts=1)
        pairs = strategy.generate([solved_trajectory])
        assert len(pairs) >= 1

    def test_skips_no_errors(self, partial_trajectory):
        strategy = ErrorCorrectionStrategy(min_attempts=5)
        pairs = strategy.generate([partial_trajectory])
        # partial trajectory has no errors, should produce no pairs
        assert len(pairs) == 0

    def test_skips_no_task(self):
        traj = SearchTrajectory(
            individuals=[
                IndividualRecord(fitness=0.0, code="bad", error="err"),
                IndividualRecord(fitness=1.0, code="good"),
            ]
        )
        strategy = ErrorCorrectionStrategy()
        pairs = strategy.generate([traj])
        assert len(pairs) == 0

    def test_name(self):
        assert ErrorCorrectionStrategy.name == "error_correction"


class TestImprovementChainStrategy:
    def test_generates_from_chain(self, solved_trajectory):
        strategy = ImprovementChainStrategy(min_steps=1)
        pairs = strategy.generate([solved_trajectory])
        assert len(pairs) >= 1
        for p in pairs:
            assert p.strategy == "improvement_chain"
            assert "fitness" in p.prompt.lower() or "improve" in p.prompt.lower()

    def test_skips_short_chain(self, failed_trajectory):
        strategy = ImprovementChainStrategy(min_steps=10)
        pairs = strategy.generate([failed_trajectory])
        assert len(pairs) == 0

    def test_cumulative_pair(self, partial_trajectory):
        # partial_trajectory has a 3-step progression with 2 improvement steps
        strategy = ImprovementChainStrategy(min_steps=2)
        pairs = strategy.generate([partial_trajectory])
        cumulative = [p for p in pairs if p.metadata.get("type") == "cumulative"]
        assert len(cumulative) >= 1

    def test_max_steps_limit(self, solved_trajectory):
        strategy = ImprovementChainStrategy(min_steps=1, max_steps=1)
        pairs = strategy.generate([solved_trajectory])
        step_pairs = [p for p in pairs if p.metadata.get("type") != "cumulative"]
        # Should have at most 1 step pair + possibly cumulative
        assert len(step_pairs) <= 1

    def test_quality_bounded(self, solved_trajectory):
        strategy = ImprovementChainStrategy(min_steps=1)
        pairs = strategy.generate([solved_trajectory])
        for p in pairs:
            assert 0.0 <= p.quality_score <= 1.0

    def test_name(self):
        assert ImprovementChainStrategy.name == "improvement_chain"


class TestHindsightRelabelStrategy:
    def test_generates_from_partial(self, partial_trajectory):
        strategy = HindsightRelabelStrategy(fitness_threshold=0.1)
        pairs = strategy.generate([partial_trajectory])
        assert len(pairs) >= 1
        for p in pairs:
            assert p.strategy == "hindsight_relabel"
            assert p.metadata.get("relabeled") is True

    def test_skips_solved(self, solved_trajectory):
        strategy = HindsightRelabelStrategy()
        pairs = strategy.generate([solved_trajectory])
        assert len(pairs) == 0

    def test_skips_below_threshold(self, partial_trajectory):
        strategy = HindsightRelabelStrategy(fitness_threshold=0.99)
        pairs = strategy.generate([partial_trajectory])
        assert len(pairs) == 0

    def test_relax_task_high_fitness(self):
        desc = HindsightRelabelStrategy._relax_task("Sort a list", 0.8)
        assert "partial" in desc.lower()

    def test_relax_task_medium_fitness(self):
        desc = HindsightRelabelStrategy._relax_task("Sort a list", 0.5)
        assert "basic" in desc.lower()

    def test_relax_task_low_fitness(self):
        desc = HindsightRelabelStrategy._relax_task("Sort a list", 0.2)
        assert "sketch" in desc.lower() or "outline" in desc.lower()

    def test_relax_strips_leading_verb(self):
        desc = HindsightRelabelStrategy._relax_task("Write a function", 0.8)
        assert not desc.lower().startswith("write")

    def test_relax_strips_implement(self):
        desc = HindsightRelabelStrategy._relax_task("Implement sorting", 0.5)
        assert "implement" not in desc.lower().split(":")[-1].strip().split()[0].lower()

    def test_name(self):
        assert HindsightRelabelStrategy.name == "hindsight_relabel"


class TestCrossoverPairsStrategy:
    def test_generates_from_crossover(self, trajectory_with_crossover):
        strategy = CrossoverPairsStrategy()
        pairs = strategy.generate([trajectory_with_crossover])
        assert len(pairs) >= 1
        for p in pairs:
            assert p.strategy == "crossover_pairs"
            assert "Solution A" in p.prompt
            assert "Solution B" in p.prompt

    def test_skips_no_crossover(self, partial_trajectory):
        # partial_trajectory individuals don't have 2 parents
        strategy = CrossoverPairsStrategy()
        pairs = strategy.generate([partial_trajectory])
        assert len(pairs) == 0

    def test_quality_with_improvement(self, trajectory_with_crossover):
        strategy = CrossoverPairsStrategy()
        pairs = strategy.generate([trajectory_with_crossover])
        assert len(pairs) >= 1
        # Child (0.9) > best parent (0.6), so quality = fitness * 1.0
        assert pairs[0].quality_score > 0

    def test_name(self):
        assert CrossoverPairsStrategy.name == "crossover_pairs"

    def test_skips_empty_code(self):
        task = TaskSpec(task_id="t", description="d")
        parent_a = IndividualRecord(individual_id="pa", code="", fitness=0.5)
        parent_b = IndividualRecord(individual_id="pb", code="code", fitness=0.5)
        child = IndividualRecord(
            individual_id="ch", code="code",
            fitness=0.9, parent_ids=["pa", "pb"], operator="crossover",
        )
        traj = SearchTrajectory(task=task, individuals=[parent_a, parent_b, child])
        strategy = CrossoverPairsStrategy()
        pairs = strategy.generate([traj])
        assert len(pairs) == 0


class TestPatternDescriptionStrategy:
    def test_generates_patterns(self, solved_trajectory):
        strategy = PatternDescriptionStrategy()
        pairs = strategy.generate([solved_trajectory])
        assert len(pairs) >= 1
        for p in pairs:
            assert p.strategy == "pattern_description"
            assert "Approach" in p.completion
            assert "Implementation" in p.completion

    def test_skips_low_fitness(self, failed_trajectory):
        strategy = PatternDescriptionStrategy()
        pairs = strategy.generate([failed_trajectory])
        assert len(pairs) == 0

    def test_includes_operators(self, solved_trajectory):
        strategy = PatternDescriptionStrategy()
        pairs = strategy.generate([solved_trajectory])
        assert len(pairs) >= 1
        # Pattern description should mention operators
        pattern = pairs[0].metadata.get("pattern", "")
        assert len(pattern) > 0

    def test_name(self):
        assert PatternDescriptionStrategy.name == "pattern_description"


class TestTrainingPair:
    def test_total_tokens(self):
        pair = TrainingPair(prompt_tokens=10, completion_tokens=20)
        assert pair.total_tokens == 30

    def test_to_dict_from_dict(self, sample_training_pairs):
        pair = sample_training_pairs[0]
        d = pair.to_dict()
        restored = TrainingPair.from_dict(d)
        assert restored.pair_id == pair.pair_id
        assert restored.strategy == pair.strategy
        assert restored.quality_score == pair.quality_score
        assert restored.prompt_tokens == pair.prompt_tokens

    def test_defaults(self):
        pair = TrainingPair()
        assert pair.strategy == ""
        assert pair.quality_score == 0.0
        assert pair.prompt_tokens == 0
        assert pair.total_tokens == 0


class TestSynthesizer:
    def test_register_and_synthesize(self, all_trajectories):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
        synthesizer.register_strategy(ErrorCorrectionStrategy(), 0.8)

        pairs = synthesizer.synthesize(all_trajectories)
        assert len(pairs) >= 1
        assert len(synthesizer.pairs) == len(pairs)

    def test_strategy_names(self):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
        assert "direct_solution" in synthesizer.strategy_names

    def test_weight_scaling(self, solved_trajectory):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(min_fitness=0.5), 0.5)
        pairs = synthesizer.synthesize([solved_trajectory])
        for p in pairs:
            # Quality should be scaled by weight
            assert p.quality_score <= 1.0

    def test_zero_weight_excluded(self, solved_trajectory):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(), 0.0)
        pairs = synthesizer.synthesize([solved_trajectory])
        assert len(pairs) == 0

    def test_summary(self, all_trajectories):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
        synthesizer.synthesize(all_trajectories)
        s = synthesizer.summary()
        assert "total_pairs" in s
        assert "by_strategy" in s

    def test_multiple_strategies(self, all_trajectories):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
        synthesizer.register_strategy(ImprovementChainStrategy(min_steps=1), 0.7)
        synthesizer.register_strategy(HindsightRelabelStrategy(fitness_threshold=0.1), 0.5)

        pairs = synthesizer.synthesize(all_trajectories)
        strategies_seen = set(p.strategy for p in pairs)
        assert len(strategies_seen) >= 1

    def test_empty_trajectories(self):
        synthesizer = Synthesizer()
        synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
        pairs = synthesizer.synthesize([])
        assert len(pairs) == 0

    def test_strategies_property(self):
        synthesizer = Synthesizer()
        s = DirectSolutionStrategy()
        synthesizer.register_strategy(s, 1.0)
        assert len(synthesizer.strategies) == 1
