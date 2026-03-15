"""Tests for TaskExecutor, MockLLMClient, create_llm_client, and related classes."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from src.core.executor import (
    Task,
    TaskResult,
    TaskExecutor,
    MockLLMClient,
    create_llm_client,
)


# ===================================================================
# MockLLMClient
# ===================================================================

class TestMockLLMClient:
    def test_default_response(self) -> None:
        client = MockLLMClient(default_response="hello")
        assert client.generate("prompt") == "hello"

    def test_queued_response(self) -> None:
        client = MockLLMClient()
        client.add_response("first")
        client.add_response("second")
        assert client.generate("p1") == "first"
        assert client.generate("p2") == "second"

    def test_add_responses_bulk(self) -> None:
        client = MockLLMClient()
        client.add_responses(["a", "b", "c"])
        assert client.generate("p") == "a"
        assert client.generate("p") == "b"
        assert client.generate("p") == "c"

    def test_auto_respond_math(self) -> None:
        client = MockLLMClient()
        resp = client.generate("What is 2 + 2?")
        assert "answer is" in resp.lower()

    def test_auto_respond_deliberation(self) -> None:
        client = MockLLMClient()
        resp = client.generate("Should we deliberate about modifications?")
        assert "modify" in resp.lower() or "action" in resp.lower()

    def test_auto_respond_generic(self) -> None:
        client = MockLLMClient()
        resp = client.generate("Tell me about cats")
        assert len(resp) > 0

    def test_call_count(self) -> None:
        client = MockLLMClient(default_response="x")
        assert client.call_count == 0
        client.generate("p1")
        client.generate("p2")
        assert client.call_count == 2

    def test_call_log(self) -> None:
        client = MockLLMClient(default_response="x")
        client.generate("prompt1", system_prompt="sys1")
        client.generate("prompt2")
        log = client.call_log
        assert len(log) == 2
        assert log[0]["prompt"] == "prompt1"
        assert log[0]["system_prompt"] == "sys1"
        assert log[1]["prompt"] == "prompt2"

    def test_queued_takes_precedence(self) -> None:
        client = MockLLMClient(default_response="default")
        client.add_response("queued")
        assert client.generate("p") == "queued"
        assert client.generate("p") == "default"


# ===================================================================
# create_llm_client
# ===================================================================

class TestCreateLLMClient:
    def test_create_mock(self) -> None:
        client = create_llm_client("mock")
        assert isinstance(client, MockLLMClient)

    def test_create_mock_with_default_response(self) -> None:
        client = create_llm_client("mock", default_response="test")
        assert isinstance(client, MockLLMClient)
        assert client.generate("p") == "test"

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client("fake_provider")

    def test_create_openai_without_package_raises(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(RuntimeError, match="Failed to create OpenAI client"):
                create_llm_client("openai")

    def test_create_anthropic_without_package_raises(self) -> None:
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(RuntimeError, match="Failed to create Anthropic client"):
                create_llm_client("anthropic")


# ===================================================================
# TaskExecutor
# ===================================================================

class TestTaskExecutor:
    def test_execute_single_task(self) -> None:
        mock = MockLLMClient(default_response="The answer is: 4")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What is 2+2?", expected_answer="4")
        result = executor.execute(task)
        assert result.correct is True
        assert result.extracted_answer == "4"
        assert result.latency > 0

    def test_execute_with_system_prompt(self) -> None:
        mock = MockLLMClient(default_response="The answer is: 42")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What is 6*7?", expected_answer="42")
        result = executor.execute(task, system_prompt="You are a calculator.")
        assert result.correct is True
        assert mock.call_log[0]["system_prompt"] == "You are a calculator."

    def test_execute_incorrect_answer(self) -> None:
        mock = MockLLMClient(default_response="The answer is: 99")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What is 2+2?", expected_answer="4")
        result = executor.execute(task)
        assert result.correct is False

    def test_execute_with_examples(self) -> None:
        mock = MockLLMClient(default_response="The answer is: 10")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What is 5+5?", expected_answer="10")
        examples = [{"question": "1+1?", "answer": "2"}]
        result = executor.execute(task, few_shot_examples=examples)
        assert result.correct is True
        # Check examples were included in prompt
        assert "1+1?" in mock.call_log[0]["prompt"]

    def test_execute_with_different_reasoning_modes(self) -> None:
        mock = MockLLMClient(default_response="The answer is: 42")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="x", expected_answer="42")

        for mode in ["cot", "code", "decompose", "direct"]:
            result = executor.execute(task, reasoning_mode=mode)
            assert result.reasoning_mode == mode

    def test_execute_handles_exception(self) -> None:
        class BrokenLLM:
            def generate(self, prompt: str, **kwargs: Any) -> str:
                raise RuntimeError("LLM is down")

        executor = TaskExecutor(BrokenLLM())
        task = Task(task_id="t1", question="x", expected_answer="y")
        result = executor.execute(task)
        assert result.correct is False
        assert "LLM is down" in result.error

    def test_execute_batch(self) -> None:
        mock = MockLLMClient(default_response="The answer is: 42")
        executor = TaskExecutor(mock)
        tasks = [
            Task(task_id=f"t{i}", question=f"What is {i}?", expected_answer="42")
            for i in range(5)
        ]
        results = executor.execute_batch(tasks)
        assert len(results) == 5
        assert all(isinstance(r, TaskResult) for r in results)

    def test_execute_validation(self) -> None:
        mock = MockLLMClient(default_response="The answer is: 42")
        executor = TaskExecutor(mock)
        tasks = [
            Task(task_id="t1", question="q1", expected_answer="42"),
            Task(task_id="t2", question="q2", expected_answer="42"),
        ]
        results = executor.execute_validation(tasks)
        assert len(results) == 2

    def test_build_prompt_cot(self) -> None:
        mock = MockLLMClient(default_response="x")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What?")
        prompt = executor._build_prompt(task, [], "cot")
        assert "step by step" in prompt.lower()

    def test_build_prompt_code(self) -> None:
        mock = MockLLMClient(default_response="x")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What?")
        prompt = executor._build_prompt(task, [], "code")
        assert "code" in prompt.lower()

    def test_build_prompt_decompose(self) -> None:
        mock = MockLLMClient(default_response="x")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What?")
        prompt = executor._build_prompt(task, [], "decompose")
        assert "break" in prompt.lower() or "smaller" in prompt.lower()

    def test_build_prompt_direct(self) -> None:
        mock = MockLLMClient(default_response="x")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What?")
        prompt = executor._build_prompt(task, [], "direct")
        assert "directly" in prompt.lower()

    def test_build_prompt_with_examples(self) -> None:
        mock = MockLLMClient(default_response="x")
        executor = TaskExecutor(mock)
        task = Task(task_id="t1", question="What?")
        examples = [{"question": "Q1", "answer": "A1"}]
        prompt = executor._build_prompt(task, examples, "cot")
        assert "Q1" in prompt
        assert "A1" in prompt


class TestExtractAnswer:
    def test_extract_with_the_answer_is(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._extract_answer("blah blah\nThe answer is: 42") == "42"

    def test_extract_with_answer_colon(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._extract_answer("Answer: 7") == "7"

    def test_extract_with_equals(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._extract_answer("x = 5") == "5"

    def test_extract_fallback_last_line(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._extract_answer("line1\nline2\n42") == "42"

    def test_extract_empty_response(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._extract_answer("") == ""

    def test_extract_strips_trailing_period(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        result = executor._extract_answer("The answer is: 42.")
        assert result == "42"


class TestCheckAnswer:
    def test_exact_match(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._check_answer("42", "42") is True

    def test_case_insensitive(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._check_answer("Yes", "yes") is True

    def test_numeric_match(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._check_answer("42.0", "42") is True

    def test_substring_match(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._check_answer("the result is 42", "42") is True

    def test_empty_extracted(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._check_answer("", "42") is False

    def test_empty_expected(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._check_answer("42", "") is False

    def test_no_match(self) -> None:
        executor = TaskExecutor(MockLLMClient())
        assert executor._check_answer("99", "42") is False
