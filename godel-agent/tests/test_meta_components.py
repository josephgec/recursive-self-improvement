"""Tests for meta components: FewShotSelector, ReasoningStrategy, DefaultPromptStrategy, ComponentRegistry."""

from __future__ import annotations

import random
from typing import Any

import pytest

from src.core.executor import Task, TaskResult
from src.meta.few_shot_selector import FewShotSelector
from src.meta.reasoning_strategy import ReasoningStrategy
from src.meta.prompt_strategy import DefaultPromptStrategy
from src.meta.registry import ComponentRegistry


# ===================================================================
# FewShotSelector
# ===================================================================

class TestFewShotSelector:
    def _make_pool(self) -> list[dict[str, Any]]:
        return [
            {"question": "What is 2+2?", "answer": "4", "category": "arithmetic", "domain": "math"},
            {"question": "What is 3*5?", "answer": "15", "category": "arithmetic", "domain": "math"},
            {"question": "Solve x+1=3", "answer": "2", "category": "algebra", "domain": "math"},
            {"question": "len('hi')", "answer": "2", "category": "strings", "domain": "code"},
            {"question": "print(1+1)", "answer": "2", "category": "arithmetic", "domain": "code"},
            {"question": "Reverse 'abc'", "answer": "cba", "category": "strings", "domain": "code"},
        ]

    def test_empty_pool_returns_empty(self) -> None:
        selector = FewShotSelector()
        task = Task(task_id="t1", question="x", category="arithmetic", domain="math")
        result = selector.select(task, [], n=3)
        assert result == []

    def test_select_n_examples(self) -> None:
        selector = FewShotSelector()
        pool = self._make_pool()
        task = Task(task_id="t1", question="x", category="arithmetic", domain="math")
        result = selector.select(task, pool, n=2)
        assert len(result) == 2

    def test_prefers_same_category(self) -> None:
        random.seed(42)
        selector = FewShotSelector()
        pool = self._make_pool()
        task = Task(task_id="t1", question="x", category="arithmetic", domain="math")
        result = selector.select(task, pool, n=2)
        # With 2 arithmetic math examples available, both should be from arithmetic
        categories = [ex.get("category") for ex in result]
        assert all(c == "arithmetic" for c in categories)

    def test_falls_back_to_domain(self) -> None:
        random.seed(42)
        selector = FewShotSelector()
        pool = self._make_pool()
        task = Task(task_id="t1", question="x", category="algebra", domain="math")
        result = selector.select(task, pool, n=3)
        # Should have the algebra example plus math domain fill
        assert len(result) == 3
        domains = [ex.get("domain") for ex in result]
        # The algebra one is definitely there
        categories = [ex.get("category") for ex in result]
        assert "algebra" in categories

    def test_falls_back_to_random(self) -> None:
        random.seed(42)
        selector = FewShotSelector()
        # Pool with only one domain entry
        pool = [
            {"question": "q1", "answer": "a1", "category": "x", "domain": "y"},
            {"question": "q2", "answer": "a2", "category": "x", "domain": "y"},
        ]
        task = Task(task_id="t1", question="x", category="other", domain="other")
        result = selector.select(task, pool, n=2)
        assert len(result) == 2

    def test_n_capped_to_pool_size(self) -> None:
        selector = FewShotSelector()
        pool = [{"question": "q1", "answer": "a1", "category": "x", "domain": "y"}]
        task = Task(task_id="t1", question="x", category="x", domain="y")
        result = selector.select(task, pool, n=5)
        assert len(result) == 1

    def test_no_category_on_task(self) -> None:
        selector = FewShotSelector()
        pool = self._make_pool()
        task = Task(task_id="t1", question="x", category="", domain="math")
        result = selector.select(task, pool, n=3)
        assert len(result) == 3


# ===================================================================
# ReasoningStrategy
# ===================================================================

class TestReasoningStrategy:
    def test_default_returns_cot(self) -> None:
        strategy = ReasoningStrategy()
        task = Task(task_id="t1", question="x")
        assert strategy.choose(task) == "cot"

    def test_returns_cot_with_results(self) -> None:
        strategy = ReasoningStrategy()
        task = Task(task_id="t1", question="x")
        results = [
            TaskResult(task=task, correct=True),
            TaskResult(task=task, correct=False),
        ]
        assert strategy.choose(task, recent_results=results) == "cot"

    def test_returns_cot_with_none_results(self) -> None:
        strategy = ReasoningStrategy()
        task = Task(task_id="t1", question="x")
        assert strategy.choose(task, recent_results=None) == "cot"


# ===================================================================
# DefaultPromptStrategy
# ===================================================================

class TestDefaultPromptStrategy:
    def test_default_system_prompt(self) -> None:
        strategy = DefaultPromptStrategy()
        assert "problem solver" in strategy.system_prompt.lower()

    def test_custom_system_prompt(self) -> None:
        strategy = DefaultPromptStrategy(system_prompt="Custom prompt.")
        assert strategy.system_prompt == "Custom prompt."

    def test_prepare_prompt_without_examples(self) -> None:
        strategy = DefaultPromptStrategy()
        task = Task(task_id="t1", question="What is 2+2?")
        prompt = strategy.prepare_prompt(task, [])
        assert "What is 2+2?" in prompt
        assert "The answer is:" in prompt

    def test_prepare_prompt_with_examples(self) -> None:
        strategy = DefaultPromptStrategy()
        task = Task(task_id="t1", question="What is 3+3?")
        examples = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
        ]
        prompt = strategy.prepare_prompt(task, examples)
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "1+1" in prompt
        assert "What is 3+3?" in prompt

    def test_select_examples_empty_pool(self) -> None:
        strategy = DefaultPromptStrategy()
        task = Task(task_id="t1", question="x", category="arithmetic")
        result = strategy.select_examples(task, [], 3)
        assert result == []

    def test_select_examples_prefers_category(self) -> None:
        random.seed(42)
        strategy = DefaultPromptStrategy()
        task = Task(task_id="t1", question="x", category="arithmetic")
        pool = [
            {"question": "q1", "answer": "a1", "category": "arithmetic"},
            {"question": "q2", "answer": "a2", "category": "arithmetic"},
            {"question": "q3", "answer": "a3", "category": "arithmetic"},
            {"question": "q4", "answer": "a4", "category": "algebra"},
        ]
        result = strategy.select_examples(task, pool, 3)
        assert len(result) == 3
        assert all(ex.get("category") == "arithmetic" for ex in result)

    def test_select_examples_mixes_when_not_enough_category(self) -> None:
        random.seed(42)
        strategy = DefaultPromptStrategy()
        task = Task(task_id="t1", question="x", category="algebra")
        pool = [
            {"question": "q1", "answer": "a1", "category": "algebra"},
            {"question": "q2", "answer": "a2", "category": "arithmetic"},
            {"question": "q3", "answer": "a3", "category": "geometry"},
        ]
        result = strategy.select_examples(task, pool, 3)
        assert len(result) == 3

    def test_choose_reasoning_mode_always_cot(self) -> None:
        strategy = DefaultPromptStrategy()
        task = Task(task_id="t1", question="x")
        assert strategy.choose_reasoning_mode(task, []) == "cot"

    def test_learn_from_result(self) -> None:
        strategy = DefaultPromptStrategy()
        task = Task(task_id="t1", question="x")
        result = TaskResult(task=task, correct=True, response="ok")
        strategy.learn_from_result(result)
        assert len(strategy.learning_buffer) == 1

    def test_learning_buffer_capped(self) -> None:
        strategy = DefaultPromptStrategy(num_examples=3)
        strategy._max_buffer = 5
        task = Task(task_id="t1", question="x")
        for i in range(10):
            strategy.learn_from_result(TaskResult(task=task, correct=True))
        assert len(strategy.learning_buffer) == 5

    def test_learning_buffer_property_returns_copy(self) -> None:
        strategy = DefaultPromptStrategy()
        task = Task(task_id="t1", question="x")
        strategy.learn_from_result(TaskResult(task=task, correct=True))
        buf = strategy.learning_buffer
        buf.clear()
        assert len(strategy.learning_buffer) == 1  # Original unaffected


# ===================================================================
# ComponentRegistry (additional coverage)
# ===================================================================

class TestComponentRegistryExtra:
    def test_register_and_get(self) -> None:
        reg = ComponentRegistry()
        reg.register("test", 42)
        assert reg.get("test") == 42

    def test_get_missing_raises(self) -> None:
        reg = ComponentRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_list_components(self) -> None:
        reg = ComponentRegistry()
        reg.register("a", 1)
        reg.register("b", 2)
        assert set(reg.list_components()) == {"a", "b"}

    def test_has(self) -> None:
        reg = ComponentRegistry()
        reg.register("x", 10)
        assert reg.has("x") is True
        assert reg.has("y") is False

    def test_replace(self) -> None:
        reg = ComponentRegistry()
        reg.register("x", 10)
        old = reg.replace("x", 20)
        assert old == 10
        assert reg.get("x") == 20

    def test_replace_new_key(self) -> None:
        reg = ComponentRegistry()
        old = reg.replace("new", 5)
        assert old is None
        assert reg.get("new") == 5

    def test_contains(self) -> None:
        reg = ComponentRegistry()
        reg.register("x", 1)
        assert "x" in reg
        assert "y" not in reg

    def test_len(self) -> None:
        reg = ComponentRegistry()
        assert len(reg) == 0
        reg.register("a", 1)
        reg.register("b", 2)
        assert len(reg) == 2
