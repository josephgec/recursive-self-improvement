"""Tests for DepthController."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.recursion.depth_controller import DepthController, BudgetExhaustedError


class TestCanRecurse:
    def test_can_recurse_at_zero(self):
        dc = DepthController(max_depth=3)
        assert dc.can_recurse(0)

    def test_cannot_recurse_at_max(self):
        dc = DepthController(max_depth=3)
        assert not dc.can_recurse(3)

    def test_cannot_recurse_beyond_max(self):
        dc = DepthController(max_depth=2)
        assert not dc.can_recurse(5)

    def test_can_recurse_default_depth(self):
        dc = DepthController(max_depth=3)
        dc._current_depth = 1
        assert dc.can_recurse()

    def test_sub_query_limit(self):
        dc = DepthController(max_depth=5, max_sub_queries=2)
        dc._sub_query_counts[0] = 2
        assert not dc.can_recurse(0)


class TestRegisterCall:
    def test_register_call(self):
        dc = DepthController()
        dc.register_call(depth=0)
        assert dc._total_calls == 1

    def test_register_multiple(self):
        dc = DepthController()
        dc.register_call()
        dc.register_call()
        assert dc._total_calls == 2

    def test_register_with_depth(self):
        dc = DepthController()
        dc.register_call(depth=2)
        assert dc._current_depth == 2


class TestRegisterSubQuery:
    def test_register_sub_query(self):
        dc = DepthController(max_depth=3)
        child_depth = dc.register_sub_query(parent_depth=0)
        assert child_depth == 1

    def test_register_increments_count(self):
        dc = DepthController(max_depth=3)
        dc.register_sub_query(parent_depth=0)
        assert dc._sub_query_counts[0] == 1

    def test_budget_exhausted_error(self):
        dc = DepthController(max_depth=1)
        with pytest.raises(BudgetExhaustedError):
            dc.register_sub_query(parent_depth=1)

    def test_sub_query_limit_error(self):
        dc = DepthController(max_depth=5, max_sub_queries=2)
        dc.register_sub_query(parent_depth=0)
        dc.register_sub_query(parent_depth=0)
        with pytest.raises(BudgetExhaustedError):
            dc.register_sub_query(parent_depth=0)


class TestRemainingBudget:
    def test_full_budget(self):
        dc = DepthController(max_sub_queries=5)
        assert dc.remaining_budget(0) == 5

    def test_partial_budget(self):
        dc = DepthController(max_sub_queries=5)
        dc.register_sub_query(0)
        dc.register_sub_query(0)
        assert dc.remaining_budget(0) == 3

    def test_exhausted_budget(self):
        dc = DepthController(max_sub_queries=1, max_depth=5)
        dc.register_sub_query(0)
        assert dc.remaining_budget(0) == 0

    def test_default_depth(self):
        dc = DepthController(max_sub_queries=5)
        dc._current_depth = 1
        assert dc.remaining_budget() == 5


class TestAllocateSubBudget:
    def test_allocate(self):
        dc = DepthController(budget_fraction=0.5)
        assert dc.allocate_sub_budget(10) == 5

    def test_allocate_minimum(self):
        dc = DepthController(budget_fraction=0.1)
        assert dc.allocate_sub_budget(1) == 1

    def test_allocate_fraction(self):
        dc = DepthController(budget_fraction=0.3)
        assert dc.allocate_sub_budget(10) == 3
