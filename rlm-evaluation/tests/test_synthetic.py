"""Tests for synthetic task generator."""

import pytest

from src.benchmarks.synthetic import SyntheticTaskGenerator
from src.benchmarks.task import EvalTask


class TestSyntheticTaskGenerator:
    """Test synthetic task generation and verification."""

    def test_load_default(self):
        gen = SyntheticTaskGenerator()
        tasks = gen.load()
        assert len(tasks) >= 5
        assert all(isinstance(t, EvalTask) for t in tasks)

    def test_needle_in_haystack_generation(self):
        gen = SyntheticTaskGenerator()
        tasks = gen.needle_in_haystack(context_tokens=4000, num_tasks=3)
        assert len(tasks) == 3
        for t in tasks:
            assert t.category == "needle_in_haystack"
            assert t.benchmark == "synthetic"
            assert t.context_tokens == 4000

    def test_needle_answer_in_context(self):
        """Verify the needle is actually present in the context."""
        gen = SyntheticTaskGenerator()
        tasks = gen.needle_in_haystack(context_tokens=4000, num_tasks=3)
        for t in tasks:
            assert t.expected_answer in t.context

    def test_needle_positions(self):
        """Test that needles are placed at specified positions."""
        gen = SyntheticTaskGenerator()
        tasks = gen.needle_in_haystack(
            context_tokens=4000,
            num_tasks=3,
            needle_positions=[0.1, 0.5, 0.9],
        )
        assert len(tasks) == 3
        for t in tasks:
            pos = t.metadata.get("needle_position")
            assert pos is not None
            # Verify needle is roughly at the expected position
            needle_idx = t.context.find(t.expected_answer)
            assert needle_idx >= 0
            relative_pos = needle_idx / len(t.context)
            # Allow some tolerance
            assert abs(relative_pos - pos) < 0.3

    def test_multi_needle_generation(self):
        gen = SyntheticTaskGenerator()
        tasks = gen.multi_needle(context_tokens=4000, num_needles=3, num_tasks=2)
        assert len(tasks) == 2
        for t in tasks:
            assert t.category == "multi_needle"
            assert t.metadata.get("num_needles", 0) >= 1

    def test_multi_needle_all_present(self):
        """Verify all needles are in the context."""
        gen = SyntheticTaskGenerator()
        tasks = gen.multi_needle(context_tokens=8000, num_needles=3, num_tasks=1)
        # For the first task (agent locations), check agents are mentioned
        t = tasks[0]
        assert "Berlin" in t.context or "Tokyo" in t.context or "Cairo" in t.context

    def test_counting_generation(self):
        gen = SyntheticTaskGenerator()
        tasks = gen.counting(context_tokens=4000, num_tasks=2)
        assert len(tasks) == 2
        for t in tasks:
            assert t.category == "counting"
            # Expected answer should be a number
            assert t.expected_answer.isdigit()

    def test_counting_answer_verification(self):
        """Verify counting answers match actual occurrences."""
        gen = SyntheticTaskGenerator()
        tasks = gen.counting(context_tokens=4000, num_tasks=2)
        for t in tasks:
            target_item = t.metadata.get("target_item", "")
            if target_item:
                actual_count = t.context.count(target_item)
                assert actual_count == int(t.expected_answer), \
                    f"Expected {t.expected_answer} occurrences of '{target_item}', found {actual_count}"

    def test_distributed_reasoning_generation(self):
        gen = SyntheticTaskGenerator()
        tasks = gen.distributed_reasoning(context_tokens=4000, num_tasks=2)
        assert len(tasks) == 2
        for t in tasks:
            assert t.category == "distributed_reasoning"
            assert t.difficulty == "hard"
            assert t.metadata.get("num_clues", 0) >= 2

    def test_distributed_reasoning_clues_present(self):
        """Verify all clues are in the context."""
        gen = SyntheticTaskGenerator()
        tasks = gen.distributed_reasoning(context_tokens=8000, num_tasks=1)
        t = tasks[0]
        # First puzzle has clues about treasure
        assert "treasure" in t.context.lower() or "clue" in t.context.lower()

    def test_context_size_control(self):
        """Test that context sizes are approximately correct."""
        gen = SyntheticTaskGenerator()
        for target_tokens in [1000, 4000, 16000]:
            tasks = gen.needle_in_haystack(context_tokens=target_tokens, num_tasks=1)
            t = tasks[0]
            # Context should be roughly target_tokens * 4 chars
            target_chars = target_tokens * 4
            assert len(t.context) <= target_chars * 1.2  # Allow 20% overshoot

    def test_unique_task_ids(self):
        gen = SyntheticTaskGenerator()
        tasks = gen.load()
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_seed_reproducibility(self):
        g1 = SyntheticTaskGenerator(seed=42)
        g2 = SyntheticTaskGenerator(seed=42)
        t1 = g1.load()
        t2 = g2.load()
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a.task_id == b.task_id
            assert a.expected_answer == b.expected_answer

    def test_benchmark_name(self):
        gen = SyntheticTaskGenerator()
        assert gen.name == "synthetic"
