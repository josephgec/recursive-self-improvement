"""Tests for the code generator (mock LLM, no API keys needed)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.code_generator import SymCodeGenerator, GenerationResult
from src.pipeline.prompts import PromptManager
from src.pipeline.router import TaskType


class TestSymCodeGenerator:
    """Test code generation with mock LLM."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.generator = SymCodeGenerator(
            mock=True,
            cache_dir=self.tmpdir,
        )

    def test_generate_returns_code(self):
        result = self.generator.generate("What is 2 + 3?")
        assert isinstance(result, GenerationResult)
        assert "answer" in result.code
        assert result.raw_response  # non-empty
        assert "```python" in result.raw_response

    def test_generate_with_task_type(self):
        result = self.generator.generate(
            "Solve x^2 - 4 = 0", task_type=TaskType.ALGEBRA
        )
        assert isinstance(result, GenerationResult)
        assert result.code  # non-empty

    def test_code_extraction(self):
        """Verify code is extracted from fenced blocks."""
        result = self.generator.generate("What is 1+1?")
        # Mock always returns code in ```python...``` block
        # The parser should extract just the code
        assert "```" not in result.code

    def test_caching(self):
        """Second call with same prompt should return cached result."""
        result1 = self.generator.generate("What is 2 + 3?", use_cache=True)
        result2 = self.generator.generate("What is 2 + 3?", use_cache=True)
        assert result2.cached is True
        assert result1.code == result2.code

    def test_cache_bypass(self):
        """use_cache=False should not return cached result."""
        result1 = self.generator.generate("What is 2 + 3?", use_cache=True)
        result2 = self.generator.generate("What is 2 + 3?", use_cache=False)
        assert result2.cached is False

    def test_generate_correction(self):
        """Test correction generation after error."""
        result = self.generator.generate_correction(
            problem="What is 2 + 3?",
            prev_code="answer = 2 + 2",
            feedback="Wrong answer: got 4, expected 5",
            attempt=2,
            max_attempts=3,
        )
        assert isinstance(result, GenerationResult)
        assert result.code  # non-empty

    def test_mock_provider(self):
        """Mock provider should work without any API keys."""
        gen = SymCodeGenerator(mock=True, cache_dir=self.tmpdir)
        result = gen.generate("Test problem")
        assert result.code
        assert "answer" in result.code

    def test_unknown_provider_raises(self):
        """Unknown provider should raise ValueError."""
        gen = SymCodeGenerator(
            config={"model": {"provider": "unknown_provider"}},
            cache_dir=self.tmpdir,
        )
        gen.mock = False
        with pytest.raises(ValueError, match="Unknown provider"):
            gen.generate("Test problem", use_cache=False)

    def test_openai_provider_calls_client(self):
        """Test OpenAI provider path with mocked client."""
        gen = SymCodeGenerator(
            config={"model": {"provider": "openai", "name": "gpt-4o"}},
            cache_dir=self.tmpdir,
        )
        gen.mock = False
        gen.provider = "openai"

        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "```python\nanswer = 7\n```"
        response = MagicMock()
        response.choices = [choice]
        response.usage.prompt_tokens = 50
        response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = response
        gen._client = mock_client

        result = gen.generate("What is 3 + 4?", use_cache=False)
        assert "answer = 7" in result.code
        mock_client.chat.completions.create.assert_called_once()

    def test_anthropic_provider_calls_client(self):
        """Test Anthropic provider path with mocked client."""
        gen = SymCodeGenerator(
            config={"model": {"provider": "anthropic", "name": "claude-3"}},
            cache_dir=self.tmpdir,
        )
        gen.mock = False
        gen.provider = "anthropic"

        mock_client = MagicMock()
        content_block = MagicMock()
        content_block.text = "```python\nanswer = 10\n```"
        response = MagicMock()
        response.content = [content_block]
        response.usage.input_tokens = 50
        response.usage.output_tokens = 20
        mock_client.messages.create.return_value = response
        gen._client = mock_client

        result = gen.generate("What is 5 + 5?", use_cache=False)
        assert "answer = 10" in result.code
        mock_client.messages.create.assert_called_once()

    def test_cache_key_deterministic(self):
        """Same messages should produce same cache key."""
        gen = SymCodeGenerator(mock=True, cache_dir=self.tmpdir)
        msgs = [{"role": "user", "content": "test"}]
        key1 = gen._cache_key(msgs)
        key2 = gen._cache_key(msgs)
        assert key1 == key2

    def test_cache_key_different_for_different_messages(self):
        gen = SymCodeGenerator(mock=True, cache_dir=self.tmpdir)
        key1 = gen._cache_key([{"role": "user", "content": "a"}])
        key2 = gen._cache_key([{"role": "user", "content": "b"}])
        assert key1 != key2

    def test_cache_put_and_get(self):
        """Test direct cache put/get operations."""
        gen = SymCodeGenerator(mock=True, cache_dir=self.tmpdir)
        result = GenerationResult(
            code="answer = 42",
            raw_response="```python\nanswer = 42\n```",
            model="test",
        )
        gen._cache_put("testkey", result)
        cached = gen._cache_get("testkey")
        assert cached is not None
        assert cached.code == "answer = 42"
        assert cached.cached is True

    def test_cache_miss(self):
        gen = SymCodeGenerator(mock=True, cache_dir=self.tmpdir)
        assert gen._cache_get("nonexistent") is None

    def test_generate_correction_with_cache(self):
        """Test that correction results are cached."""
        gen = SymCodeGenerator(mock=True, cache_dir=self.tmpdir)
        result1 = gen.generate_correction(
            problem="What is 2 + 3?",
            prev_code="answer = 4",
            feedback="Wrong answer",
            attempt=2,
            max_attempts=3,
            use_cache=True,
        )
        result2 = gen.generate_correction(
            problem="What is 2 + 3?",
            prev_code="answer = 4",
            feedback="Wrong answer",
            attempt=2,
            max_attempts=3,
            use_cache=True,
        )
        assert result2.cached is True
        assert result1.code == result2.code

    def test_get_client_cached(self):
        """_get_client should return cached client."""
        gen = SymCodeGenerator(mock=True, cache_dir=self.tmpdir)
        sentinel = object()
        gen._client = sentinel
        assert gen._get_client() is sentinel

    def test_config_defaults(self):
        """Default config values should be set correctly."""
        gen = SymCodeGenerator(cache_dir=self.tmpdir)
        assert gen.provider == "openai"
        assert gen.model_name == "gpt-4o"
        assert gen.temperature == 0.0
        assert gen.max_tokens == 4096
