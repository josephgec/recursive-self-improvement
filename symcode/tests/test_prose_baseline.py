"""Tests for the prose chain-of-thought baseline solver."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.prose_baseline import ProseBaseline


class TestProseBaselineInit:
    """Test ProseBaseline construction with various configs."""

    def test_default_init(self):
        baseline = ProseBaseline()
        assert baseline.provider == "openai"
        assert baseline.model_name == "gpt-4o"
        assert baseline.temperature == 0.0
        assert baseline.max_tokens == 4096
        assert baseline.mock is False

    def test_mock_init(self):
        baseline = ProseBaseline(mock=True)
        assert baseline.provider == "mock"
        assert baseline.mock is True

    def test_config_override(self):
        config = {
            "model": {
                "provider": "anthropic",
                "name": "claude-3-opus",
                "temperature": 0.5,
                "max_tokens": 2048,
            }
        }
        baseline = ProseBaseline(config=config)
        assert baseline.provider == "anthropic"
        assert baseline.model_name == "claude-3-opus"
        assert baseline.temperature == 0.5
        assert baseline.max_tokens == 2048


class TestProseBaselineSolve:
    """Test the solve method with mock LLM."""

    def test_solve_mock_returns_answer(self):
        baseline = ProseBaseline(mock=True)
        answer, response = baseline.solve("What is 2 + 3?")
        assert answer is not None
        assert "42" in answer  # mock returns \boxed{42}
        assert "boxed" in response

    def test_solve_mock_response_text(self):
        baseline = ProseBaseline(mock=True)
        answer, response = baseline.solve("What is 2 + 3?")
        assert "step by step" in response.lower()

    def test_solve_extracts_from_boxed(self):
        """Verify answer is extracted from \\boxed{} in response."""
        baseline = ProseBaseline(mock=True)
        answer, response = baseline.solve("Some problem")
        assert answer is not None
        assert "\\boxed{42}" in response

    def test_solve_returns_none_when_no_answer(self):
        """If extraction fails, answer should be None."""
        baseline = ProseBaseline(mock=True)
        # Override _mock_response to return text without boxed or patterns
        baseline._mock_response = lambda msgs: "I have no idea how to solve this."
        answer, response = baseline.solve("Unsolvable problem")
        assert answer is None
        assert response == "I have no idea how to solve this."


class TestProseBaselineCallLLM:
    """Test _call_llm dispatch for different providers."""

    def test_call_llm_mock_flag_overrides_provider(self):
        """Even if provider != 'mock', mock=True should use mock."""
        config = {"model": {"provider": "openai"}}
        baseline = ProseBaseline(config=config, mock=True)
        messages = [{"role": "user", "content": "test"}]
        result = baseline._call_llm(messages)
        assert "boxed" in result

    def test_call_llm_openai_provider(self):
        """Test OpenAI provider path with mocked client."""
        baseline = ProseBaseline()
        baseline.provider = "openai"

        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "The answer is \\boxed{7}."
        response = MagicMock()
        response.choices = [choice]
        mock_client.chat.completions.create.return_value = response

        baseline._client = mock_client

        messages = [
            {"role": "system", "content": "You are a math solver."},
            {"role": "user", "content": "What is 3 + 4?"},
        ]
        result = baseline._call_llm(messages)
        assert result == "The answer is \\boxed{7}."
        mock_client.chat.completions.create.assert_called_once()

    def test_call_llm_anthropic_provider(self):
        """Test Anthropic provider path with mocked client."""
        baseline = ProseBaseline()
        baseline.provider = "anthropic"

        mock_client = MagicMock()
        content_block = MagicMock()
        content_block.text = "The answer is \\boxed{10}."
        response = MagicMock()
        response.content = [content_block]
        mock_client.messages.create.return_value = response

        baseline._client = mock_client

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 5 + 5?"},
        ]
        result = baseline._call_llm(messages)
        assert result == "The answer is \\boxed{10}."

        # Verify system message is separated
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "You are helpful."
        assert len(call_kwargs.kwargs["messages"]) == 1  # only user message

    def test_call_llm_unknown_provider_raises(self):
        baseline = ProseBaseline()
        baseline.provider = "unknown_provider"
        baseline.mock = False
        messages = [{"role": "user", "content": "test"}]
        with pytest.raises(ValueError, match="Unknown provider"):
            baseline._call_llm(messages)


class TestProseBaselineGetClient:
    """Test lazy client initialization."""

    def test_get_client_returns_cached(self):
        baseline = ProseBaseline()
        sentinel = object()
        baseline._client = sentinel
        assert baseline._get_client() is sentinel

    def test_get_client_openai(self):
        baseline = ProseBaseline()
        baseline.provider = "openai"
        with patch("src.pipeline.prose_baseline.openai", create=True) as mock_openai:
            # Simulate the import inside _get_client
            import sys
            mock_mod = MagicMock()
            with patch.dict(sys.modules, {"openai": mock_mod}):
                client = baseline._get_client()
                assert client is not None

    def test_get_client_anthropic(self):
        baseline = ProseBaseline()
        baseline.provider = "anthropic"
        import sys
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_mod}):
            client = baseline._get_client()
            assert client is not None
