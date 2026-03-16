"""Tests for the RLM Wrapper."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.pipeline.state import PipelineState, AgentCodeSnapshot, PerformanceRecord
from src.scaling.rlm_wrapper import RLMWrapper
from src.scaling.context_manager import ContextManager
from src.scaling.memory_bridge import MemoryBridge
from src.scaling.recursive_improvement import RecursiveImprover


def _make_state(code="def solve(x): return x + 1", accuracy=0.7):
    return PipelineState(
        agent_code=AgentCodeSnapshot(code=code),
        original_code=AgentCodeSnapshot(code=code),
        performance=PerformanceRecord(accuracy=accuracy, entropy=1.0),
    )


class TestEvolveWithContext:
    """Test evolve_with_context."""

    def test_evolve_returns_candidates(self):
        rlm = RLMWrapper()
        state = _make_state()
        candidates = rlm.evolve_with_context(state, n=3)

        assert len(candidates) == 3
        for c in candidates:
            assert c.operator == "rlm_evolve"
            assert c.proposed_code != ""

    def test_evolve_uses_context(self):
        ctx = ContextManager()
        rlm = RLMWrapper(context_manager=ctx)
        state = _make_state()
        rlm.evolve_with_context(state, n=2)

        assert ctx.get_context_size() > 0


class TestVerifyWithContext:
    """Test verify_with_context."""

    def test_verify_returns_result(self):
        rlm = RLMWrapper()
        state = _make_state()
        from src.outer_loop.strategy_evolver import Candidate
        candidate = Candidate(target="t", proposed_code="pass", operator="test")

        result = rlm.verify_with_context(candidate, state)

        assert "verified" in result
        assert "confidence" in result
        assert result["verified"] is True

    def test_verify_includes_reasoning(self):
        rlm = RLMWrapper()
        state = _make_state()
        from src.outer_loop.strategy_evolver import Candidate
        candidate = Candidate(target="t", proposed_code="pass")

        result = rlm.verify_with_context(candidate, state)
        assert "reasoning" in result


class TestInspectWithContext:
    """Test inspect_with_context."""

    def test_inspect_returns_assessment(self):
        rlm = RLMWrapper()
        state = _make_state()

        result = rlm.inspect_with_context(state)

        assert "assessment" in result
        assert "suggestions" in result
        assert "risk_level" in result

    def test_inspect_risk_level(self):
        rlm = RLMWrapper()
        state = _make_state()
        result = rlm.inspect_with_context(state)
        assert result["risk_level"] in ("low", "medium", "high", "critical")


class TestWrapIteration:
    """Test wrap_iteration."""

    def test_wrap_manages_session(self):
        rlm = RLMWrapper()
        state = _make_state()

        def dummy_iteration(s):
            assert rlm.session_active is True
            return "done"

        result = rlm.wrap_iteration(state, dummy_iteration)
        assert result == "done"
        assert rlm.session_active is False

    def test_wrap_saves_state_to_memory(self):
        memory = MemoryBridge()
        rlm = RLMWrapper(memory_bridge=memory)
        state = _make_state()

        rlm.wrap_iteration(state, lambda s: None)
        loaded = memory.load_state_from_repl()
        assert loaded is not None
        assert loaded.agent_code.code == state.agent_code.code


class TestContextManager:
    """Test context manager."""

    def test_load_codebase(self):
        ctx = ContextManager()
        tokens = ctx.load_codebase("def f(): pass")
        assert tokens > 0

    def test_load_dataset(self):
        ctx = ContextManager()
        tokens = ctx.load_dataset("sample data here")
        assert tokens > 0

    def test_load_history(self):
        ctx = ContextManager()
        tokens = ctx.load_history([{"iteration": 1, "accuracy": 0.7}])
        assert tokens > 0

    def test_get_context_size(self):
        ctx = ContextManager()
        ctx.load_codebase("code")
        assert ctx.get_context_size() > 0

    def test_fits_in_context(self):
        ctx = ContextManager(max_tokens=1000000)
        ctx.load_codebase("short")
        assert ctx.fits_in_context() is True

    def test_does_not_fit(self):
        ctx = ContextManager(max_tokens=1)
        ctx.load_codebase("this is longer than 4 characters")
        assert ctx.fits_in_context() is False

    def test_get_context(self):
        ctx = ContextManager()
        ctx.load_codebase("code")
        ctx.load_dataset("data")
        result = ctx.get_context()
        assert result["codebase"] == "code"
        assert result["dataset"] == "data"


class TestMemoryBridge:
    """Test memory bridge."""

    def test_save_and_load(self):
        bridge = MemoryBridge()
        state = _make_state()
        bridge.save_state_to_repl(state)
        loaded = bridge.load_state_from_repl()

        assert loaded is not None
        assert loaded.agent_code.code == state.agent_code.code

    def test_load_missing_returns_none(self):
        bridge = MemoryBridge()
        assert bridge.load_state_from_repl("nonexistent") is None

    def test_set_get(self):
        bridge = MemoryBridge()
        bridge.set("key", "value")
        assert bridge.get("key") == "value"
        assert bridge.get("missing", "default") == "default"

    def test_keys(self):
        bridge = MemoryBridge()
        bridge.set("a", 1)
        bridge.set("b", 2)
        assert set(bridge.keys()) == {"a", "b"}

    def test_clear(self):
        bridge = MemoryBridge()
        bridge.set("a", 1)
        bridge.clear()
        assert bridge.keys() == []


class TestRecursiveImprover:
    """Test recursive improver."""

    def test_improve_no_improvement(self):
        rlm = RLMWrapper()
        improver = RecursiveImprover(rlm_wrapper=rlm, max_depth=3)
        state = _make_state(accuracy=0.7)

        def no_change_iteration(s):
            return None

        result = improver.improve_within_rlm(state, no_change_iteration)
        assert result["status"] == "no_improvement"
        assert len(result["improvements"]) >= 1

    def test_improve_with_improvement(self):
        rlm = RLMWrapper()
        improver = RecursiveImprover(rlm_wrapper=rlm, max_depth=2)
        state = _make_state(accuracy=0.7)

        call_count = [0]

        def improving_iteration(s):
            call_count[0] += 1
            s.performance.accuracy += 0.05
            return None

        result = improver.improve_within_rlm(state, improving_iteration)
        assert "improvements" in result
        assert call_count[0] == 2  # recurses once

    def test_max_depth_reached(self):
        rlm = RLMWrapper()
        improver = RecursiveImprover(rlm_wrapper=rlm, max_depth=1)
        state = _make_state(accuracy=0.7)

        def improving_iteration(s):
            s.performance.accuracy += 0.05

        result = improver.improve_within_rlm(state, improving_iteration)
        assert result["depth"] == 0
        # With max_depth=1, it can run depth 0 then would try depth 1 but that is >= max_depth
        assert result["status"] == "improved"

    def test_clear_log(self):
        improver = RecursiveImprover()
        state = _make_state()
        improver.improve_within_rlm(state, lambda s: None)
        assert len(improver.improvement_log) >= 1
        improver.clear_log()
        assert len(improver.improvement_log) == 0
