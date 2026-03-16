"""Tests for rlm.completion() functional API."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.core.completion import completion, RLMCompletionAPI
from tests.conftest import MockLLM


class TestCompletion:
    def test_completion_basic(self, mock_llm, small_context):
        result = completion(
            prompt="Find the secret value",
            context=small_context,
            llm=mock_llm,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_completion_with_config(self, mock_llm, small_context):
        result = completion(
            prompt="Find something",
            context=small_context,
            llm=mock_llm,
            config={"max_iterations": 3},
        )
        assert isinstance(result, str)

    def test_completion_with_dict_context(self, mock_llm, large_context):
        result = completion(
            prompt="What is the total score?",
            context=large_context,
            llm=mock_llm,
        )
        assert isinstance(result, str)


class TestRLMCompletionAPI:
    def test_api_complete(self, mock_llm, small_context):
        api = RLMCompletionAPI(llm=mock_llm)
        result = api.complete(prompt="Find the value", context=small_context)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_api_create_session(self, mock_llm):
        api = RLMCompletionAPI(llm=mock_llm)
        session = api.create_session()
        assert session is not None
        assert session.max_iterations == 10  # default

    def test_api_create_session_custom_iterations(self, mock_llm):
        api = RLMCompletionAPI(llm=mock_llm)
        session = api.create_session(max_iterations=5)
        assert session.max_iterations == 5

    def test_api_no_llm_error(self):
        api = RLMCompletionAPI()
        with pytest.raises(ValueError, match="No LLM"):
            api.complete(prompt="test", context="test")

    def test_api_complete_batch(self, mock_llm, small_context):
        api = RLMCompletionAPI(llm=mock_llm)
        prompts = [
            {"prompt": "Find A", "context": small_context},
            {"prompt": "Find B", "context": small_context},
        ]
        results = api.complete_batch(prompts)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_api_with_override_llm(self, mock_llm, small_context):
        api = RLMCompletionAPI()
        result = api.complete(
            prompt="test", context=small_context, llm=mock_llm
        )
        assert isinstance(result, str)

    def test_api_with_config(self, mock_llm, small_context):
        api = RLMCompletionAPI(
            llm=mock_llm,
            config={"max_iterations": 3, "max_depth": 1},
        )
        result = api.complete(prompt="test", context=small_context)
        assert isinstance(result, str)
