"""Tests for RLMCodeExecutor."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.core.code_executor import RLMCodeExecutor, CodeBlockResult
from src.core.context_loader import ContextLoader, ContextMeta


@pytest.fixture
def executor_with_context():
    """Create an executor with a sample context loaded."""
    loader = ContextLoader()
    executor = RLMCodeExecutor(max_iterations=10)
    repl = executor.repl
    meta = loader.load_into_repl(
        "Line 1: Hello World\nLine 2: Secret Code ABC\nLine 3: Goodbye",
        repl,
    )
    executor.setup(meta)
    return executor


class TestExecuteBlock:
    def test_simple_print(self, executor_with_context):
        result = executor_with_context.execute_block("print('hello')")
        assert result.success
        assert "hello" in result.stdout

    def test_variable_assignment(self, executor_with_context):
        executor_with_context.execute_block("x = 42")
        assert executor_with_context.repl["x"] == 42

    def test_error_handling(self, executor_with_context):
        result = executor_with_context.execute_block("1/0")
        assert not result.success
        assert result.exception is not None
        assert "division by zero" in result.exception

    def test_iteration_tracking(self, executor_with_context):
        assert executor_with_context.iteration == 0
        executor_with_context.execute_block("x = 1")
        assert executor_with_context.iteration == 1
        executor_with_context.execute_block("x = 2")
        assert executor_with_context.iteration == 2

    def test_final_detection(self, executor_with_context):
        result = executor_with_context.execute_block('FINAL("done")')
        assert result.final_signal is not None
        assert result.final_signal.argument == "done"

    def test_final_var_detection(self, executor_with_context):
        executor_with_context.execute_block("answer = 'my answer'")
        result = executor_with_context.execute_block('FINAL_VAR("answer")')
        assert result.final_signal is not None
        assert result.final_signal.argument == "answer"

    def test_multiline_code(self, executor_with_context):
        code = "x = 10\ny = 20\nresult = x + y\nprint(result)"
        result = executor_with_context.execute_block(code)
        assert result.success
        assert "30" in result.stdout


class TestHelpers:
    def test_peek(self, executor_with_context):
        result = executor_with_context.execute_block(
            "preview = peek(0, 20)\nprint(preview)"
        )
        assert result.success
        assert "Line 1" in result.stdout

    def test_grep(self, executor_with_context):
        result = executor_with_context.execute_block(
            "matches = grep('Secret')\nprint(matches)"
        )
        assert result.success
        assert "Secret Code ABC" in result.stdout

    def test_search(self, executor_with_context):
        result = executor_with_context.execute_block(
            "matches = search('Hello')\nprint(matches)"
        )
        assert result.success
        assert "Hello" in result.stdout

    def test_chunk(self, executor_with_context):
        result = executor_with_context.execute_block(
            "chunks = chunk(20)\nprint(len(chunks))"
        )
        assert result.success

    def test_count_lines(self, executor_with_context):
        result = executor_with_context.execute_block(
            "n = count_lines()\nprint(n)"
        )
        assert result.success
        assert "3" in result.stdout


class TestExtractCodeBlocks:
    def test_fenced_python(self):
        response = "Here is the code:\n```python\nx = 42\nprint(x)\n```"
        blocks = RLMCodeExecutor.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "x = 42" in blocks[0]

    def test_fenced_no_language(self):
        response = "```\nprint('hello')\n```"
        blocks = RLMCodeExecutor.extract_code_blocks(response)
        assert len(blocks) == 1

    def test_multiple_blocks(self):
        response = "```python\nx = 1\n```\nSome text\n```python\ny = 2\n```"
        blocks = RLMCodeExecutor.extract_code_blocks(response)
        assert len(blocks) == 2

    def test_no_fences_but_code_like(self):
        response = "import os\nresult = os.getcwd()\nprint(result)"
        blocks = RLMCodeExecutor.extract_code_blocks(response)
        assert len(blocks) == 1

    def test_no_code(self):
        response = "This is just plain text with no code."
        blocks = RLMCodeExecutor.extract_code_blocks(response)
        assert len(blocks) == 0

    def test_empty_response(self):
        blocks = RLMCodeExecutor.extract_code_blocks("")
        assert len(blocks) == 0


class TestBudget:
    def test_budget_remaining(self):
        executor = RLMCodeExecutor(max_iterations=5)
        assert executor.budget_remaining() == 5
        executor.iteration = 3
        assert executor.budget_remaining() == 2

    def test_budget_exhausted(self):
        executor = RLMCodeExecutor(max_iterations=2)
        assert not executor.budget_exhausted()
        executor.iteration = 2
        assert executor.budget_exhausted()

    def test_truncation(self):
        executor = RLMCodeExecutor(max_output_lines=3)
        text = "line1\nline2\nline3\nline4\nline5"
        result = executor._truncate(text)
        assert "truncated" in result
