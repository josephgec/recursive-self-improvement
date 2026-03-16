"""Tests for RLMSession."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.core.session import RLMSession, SessionResult, TrajectoryStep
from src.recursion.depth_controller import DepthController
from tests.conftest import MockLLM, MockLLMImmediate, MockLLMNeverFinal


class TestSessionBasic:
    def test_session_produces_result(self, mock_llm, small_context):
        session = RLMSession(llm=mock_llm, max_iterations=5)
        result = session.run(query="Find the secret value", context=small_context)

        assert isinstance(result, SessionResult)
        assert result.result is not None
        assert result.total_iterations > 0

    def test_session_with_dict_context(self, mock_llm, large_context):
        session = RLMSession(llm=mock_llm, max_iterations=5)
        result = session.run(query="What is the secret field?", context=large_context)

        assert result.result is not None

    def test_session_trajectory(self, mock_llm, small_context):
        session = RLMSession(llm=mock_llm, max_iterations=5)
        result = session.run(query="Find something", context=small_context)

        assert len(result.trajectory) > 0
        for step in result.trajectory:
            assert isinstance(step, TrajectoryStep)
            assert step.iteration > 0

    def test_session_str(self, mock_llm, small_context):
        session = RLMSession(llm=mock_llm, max_iterations=5)
        result = session.run(query="Find the value", context=small_context)
        assert str(result) != ""


class TestSessionMultiIteration:
    def test_multi_iteration(self, mock_llm, small_context):
        session = RLMSession(llm=mock_llm, max_iterations=10)
        result = session.run(
            query="Find the secret value in the text",
            context=small_context,
        )

        # MockLLM should take ~3 iterations: peek, grep, FINAL
        assert result.total_iterations >= 2

    def test_trajectory_has_code(self, mock_llm, small_context):
        session = RLMSession(llm=mock_llm, max_iterations=5)
        result = session.run(query="Find something", context=small_context)

        has_code = any(step.code_blocks for step in result.trajectory)
        assert has_code


class TestSessionForcedFinal:
    def test_forced_final_on_budget(self, mock_llm_never_final, small_context):
        session = RLMSession(
            llm=mock_llm_never_final,
            max_iterations=3,
            forced_final=True,
        )
        result = session.run(query="Find something", context=small_context)

        assert result.forced_final
        assert result.result is not None
        assert result.total_iterations == 3

    def test_no_forced_final_when_disabled(self, mock_llm_never_final, small_context):
        session = RLMSession(
            llm=mock_llm_never_final,
            max_iterations=2,
            forced_final=False,
        )
        result = session.run(query="Find something", context=small_context)

        assert not result.forced_final
        assert result.result is None

    def test_forced_final_with_result_var(self, mock_llm_never_final, small_context):
        """When forced, the session should look for common variable names."""
        session = RLMSession(
            llm=mock_llm_never_final,
            max_iterations=2,
            forced_final=True,
        )
        # The MockLLMNeverFinal sets 'result' variable via peek
        result = session.run(query="test", context=small_context)
        assert result.forced_final
        assert result.result is not None


class TestSessionImmediate:
    def test_immediate_final(self, mock_llm_immediate, small_context):
        session = RLMSession(llm=mock_llm_immediate, max_iterations=5)
        result = session.run(query="Find the secret", context=small_context)

        assert result.total_iterations == 1
        assert not result.forced_final

    def test_immediate_result_value(self, mock_llm_immediate, small_context):
        session = RLMSession(llm=mock_llm_immediate, max_iterations=5)
        result = session.run(query="Find the secret", context=small_context)

        assert result.result is not None


class TestSessionDepth:
    def test_session_depth(self, mock_llm, small_context):
        session = RLMSession(llm=mock_llm, max_iterations=5, depth=0)
        result = session.run(query="test", context=small_context)
        assert result.depth == 0

    def test_session_elapsed_time(self, mock_llm, small_context):
        session = RLMSession(llm=mock_llm, max_iterations=5)
        result = session.run(query="test", context=small_context)
        assert result.elapsed_time >= 0

    def test_callable_llm(self, small_context):
        """Test that a plain callable works as an LLM."""
        def my_llm(messages, meta):
            return '```python\nFINAL("callable works")\n```'

        session = RLMSession(llm=my_llm, max_iterations=5)
        result = session.run(query="test", context=small_context)
        assert result.result is not None
        assert "callable works" in str(result.result)

    def test_invalid_llm_type(self, small_context):
        """Non-callable, no chat/complete should raise TypeError."""
        session = RLMSession(llm=42, max_iterations=5)
        with pytest.raises(TypeError):
            session.run(query="test", context=small_context)
