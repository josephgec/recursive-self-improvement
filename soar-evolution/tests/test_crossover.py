"""Tests for LLMCrossover."""

import pytest
from src.operators.crossover import LLMCrossover, default_mock_crossover
from src.population.individual import Individual


class TestLLMCrossover:
    def test_crossover(self, color_swap_task):
        parent_a = Individual(
            code="def transform(g): return [row[:] for row in g]\n",
            fitness=0.5,
            train_accuracy=0.5,
        )
        parent_b = Individual(
            code="def transform(g): return [[0]*len(g[0]) for _ in g]\n",
            fitness=0.3,
            train_accuracy=0.3,
        )
        xover = LLMCrossover()
        child = xover.crossover(parent_a, parent_b, color_swap_task)
        assert child is not None
        assert child.generation == max(parent_a.generation, parent_b.generation) + 1
        assert parent_a.individual_id in child.parent_ids
        assert parent_b.individual_id in child.parent_ids

    def test_analyze_complementarity(self):
        parent_a = Individual(
            code="def transform(g): return g\n" * 5,
            fitness=0.8,
            train_accuracy=0.8,
        )
        parent_b = Individual(
            code="def different(g): return [[0]]\n" * 5,
            fitness=0.3,
            train_accuracy=0.3,
        )
        xover = LLMCrossover()
        comp = xover.analyze_complementarity(parent_a, parent_b)
        assert "similarity" in comp
        assert "complementarity" in comp
        assert "is_complementary" in comp

    def test_complementarity_identical(self):
        ind = Individual(
            code="def transform(g): return g",
            fitness=0.5,
            train_accuracy=0.5,
        )
        xover = LLMCrossover()
        comp = xover.analyze_complementarity(ind, ind)
        assert comp["similarity"] == 1.0

    def test_crossover_with_failing_llm(self, color_swap_task):
        def failing_llm(prompt):
            raise RuntimeError("LLM failed")

        parent_a = Individual(code="def transform(g): return g", fitness=0.8)
        parent_b = Individual(code="def transform(g): return [[0]]", fitness=0.3)

        xover = LLMCrossover(llm_call=failing_llm)
        child = xover.crossover(parent_a, parent_b, color_swap_task)
        # Should fall back to better parent
        assert child.code == parent_a.code

    def test_crossover_fallback_worse_first(self, color_swap_task):
        def failing_llm(prompt):
            raise RuntimeError("LLM failed")

        parent_a = Individual(code="def transform(g): return g", fitness=0.3)
        parent_b = Individual(code="def transform(g): return [[0]]", fitness=0.8)

        xover = LLMCrossover(llm_call=failing_llm)
        child = xover.crossover(parent_a, parent_b, color_swap_task)
        assert child.code == parent_b.code

    def test_stats(self, color_swap_task):
        parent_a = Individual(code="def transform(g): return g", fitness=0.5, train_accuracy=0.5)
        parent_b = Individual(code="def transform(g): return [[0]]", fitness=0.3, train_accuracy=0.3)
        xover = LLMCrossover()
        xover.crossover(parent_a, parent_b, color_swap_task)
        assert xover.stats["attempted"] == 1
        assert xover.stats["successful"] == 1


class TestDefaultMockCrossover:
    def test_with_two_code_blocks(self):
        prompt = (
            "Combine:\n"
            "```python\ndef transform(g): return g\n```\n"
            "```python\ndef transform(g): return [[0]]\n```\n"
        )
        result = default_mock_crossover(prompt)
        assert "def transform" in result

    def test_with_one_code_block(self):
        prompt = "Combine:\n```python\ndef transform(g): return g\n```\n"
        result = default_mock_crossover(prompt)
        assert "def transform" in result

    def test_with_no_code_blocks(self):
        prompt = "Combine these programs"
        result = default_mock_crossover(prompt)
        assert "def transform" in result

    def test_with_generic_code_blocks(self):
        prompt = (
            "```\ndef transform(g): return g\n```\n"
            "```\ndef transform(g): return [[0]]\n```\n"
        )
        result = default_mock_crossover(prompt)
        assert "def transform" in result
