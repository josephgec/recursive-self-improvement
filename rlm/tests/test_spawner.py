"""Tests for SubQuerySpawner."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.recursion.spawner import SubQuerySpawner
from src.recursion.depth_controller import DepthController, BudgetExhaustedError
from tests.conftest import MockLLM


class TestSubQuerySpawner:
    def test_inject_into_repl(self):
        dc = DepthController(max_depth=3)
        spawner = SubQuerySpawner(
            depth_controller=dc,
            llm_factory=MockLLM,
            parent_depth=0,
        )
        repl: dict = {}
        spawner.inject_into_repl(repl)
        assert "rlm_sub_query" in repl
        assert callable(repl["rlm_sub_query"])

    def test_create_sub_query_function(self):
        dc = DepthController(max_depth=3)
        spawner = SubQuerySpawner(
            depth_controller=dc,
            llm_factory=MockLLM,
            parent_depth=0,
        )
        fn = spawner.create_sub_query_function()
        assert callable(fn)

    def test_spawn_session(self):
        dc = DepthController(max_depth=3, max_iterations=5)
        spawner = SubQuerySpawner(
            depth_controller=dc,
            llm_factory=MockLLM,
            parent_depth=0,
        )
        result = spawner._spawn_session(
            query="Find the secret",
            context="The secret is: ABC123",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_child_depth_incremented(self):
        dc = DepthController(max_depth=3, max_iterations=5)
        spawner = SubQuerySpawner(
            depth_controller=dc,
            llm_factory=MockLLM,
            parent_depth=0,
        )
        spawner._spawn_session(query="test", context="test context")
        assert len(spawner.sub_sessions) == 1

    def test_budget_exhaustion(self):
        dc = DepthController(max_depth=1, max_sub_queries=1, max_iterations=5)
        spawner = SubQuerySpawner(
            depth_controller=dc,
            llm_factory=MockLLM,
            parent_depth=0,
        )
        # First sub-query should work
        spawner._spawn_session(query="test", context="ctx")

        # Second should fail (max_sub_queries=1)
        with pytest.raises(BudgetExhaustedError):
            spawner._spawn_session(query="test2", context="ctx2")

    def test_depth_limit(self):
        dc = DepthController(max_depth=1)
        spawner = SubQuerySpawner(
            depth_controller=dc,
            llm_factory=MockLLM,
            parent_depth=1,  # Already at max depth
        )
        with pytest.raises(BudgetExhaustedError):
            spawner._spawn_session(query="test", context="ctx")
