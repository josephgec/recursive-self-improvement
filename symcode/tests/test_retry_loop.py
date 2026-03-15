"""Tests for the retry loop."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.code_generator import GenerationResult, SymCodeGenerator
from src.pipeline.prompts import PromptManager
from src.pipeline.router import TaskType
from src.verification.executor import SymCodeExecutor
from src.verification.retry_loop import RetryLoop


def _make_generator_with_sequence(code_sequence: list[str]) -> SymCodeGenerator:
    """Create a mock SymCodeGenerator that yields codes in order."""
    gen = SymCodeGenerator(mock=True, cache_dir=tempfile.mkdtemp())

    call_count = 0

    def _mock_generate(problem, task_type=None, use_cache=True):
        nonlocal call_count
        idx = min(call_count, len(code_sequence) - 1)
        code = code_sequence[idx]
        call_count += 1
        return GenerationResult(
            code=code,
            raw_response=f"```python\n{code}\n```",
            model="mock",
        )

    def _mock_correction(problem, prev_code, feedback, attempt=1, max_attempts=3, use_cache=True):
        nonlocal call_count
        idx = min(call_count, len(code_sequence) - 1)
        code = code_sequence[idx]
        call_count += 1
        return GenerationResult(
            code=code,
            raw_response=f"```python\n{code}\n```",
            model="mock",
        )

    gen.generate = _mock_generate
    gen.generate_correction = _mock_correction
    return gen


class TestRetryLoop:
    """Test the generate -> execute -> verify -> retry loop."""

    def test_first_attempt_success(self):
        """Correct code on first try should succeed with 1 attempt."""
        code = 'answer = 5\nprint(f"Answer: {answer}")\n'
        gen = _make_generator_with_sequence([code])
        loop = RetryLoop(generator=gen, max_retries=3)
        result = loop.solve("What is 2 + 3?", expected_answer="5")

        assert result.correct is True
        assert result.num_attempts == 1
        assert result.final_answer == "5"
        assert len(result.attempts) == 1

    def test_self_correction_on_attempt_2(self):
        """Error on attempt 1, correct on attempt 2."""
        bad_code = "answer = 1 / 0\n"  # ZeroDivisionError
        good_code = 'answer = 5\nprint(f"Answer: {answer}")\n'
        gen = _make_generator_with_sequence([bad_code, good_code])
        loop = RetryLoop(generator=gen, max_retries=3)
        result = loop.solve("What is 2 + 3?", expected_answer="5")

        assert result.correct is True
        assert result.num_attempts == 2
        assert result.final_answer == "5"
        # First attempt should have failed
        assert result.attempts[0].execution_result.success is False
        # Second attempt should have succeeded
        assert result.attempts[1].execution_result.success is True

    def test_all_attempts_fail(self):
        """All attempts produce errors -- should exhaust retries."""
        bad_code = "answer = 1 / 0\n"
        gen = _make_generator_with_sequence([bad_code, bad_code, bad_code])
        loop = RetryLoop(generator=gen, max_retries=3)
        result = loop.solve("What is 2 + 3?", expected_answer="5")

        assert result.correct is False
        assert result.num_attempts == 3
        assert len(result.attempts) == 3

    def test_wrong_answer_triggers_retry(self):
        """Code runs but gives wrong answer, should retry."""
        wrong_code = 'answer = 4\nprint(f"Answer: {answer}")\n'
        correct_code = 'answer = 5\nprint(f"Answer: {answer}")\n'
        gen = _make_generator_with_sequence([wrong_code, correct_code])
        loop = RetryLoop(generator=gen, max_retries=3)
        result = loop.solve("What is 2 + 3?", expected_answer="5")

        assert result.correct is True
        assert result.num_attempts == 2
        assert result.final_answer == "5"
        # First attempt had wrong answer
        assert result.attempts[0].answer_correct is False
        assert result.attempts[0].extracted_answer == "4"

    def test_no_expected_answer(self):
        """Without expected answer, first successful execution is accepted."""
        code = 'answer = 99\nprint(f"Answer: {answer}")\n'
        gen = _make_generator_with_sequence([code])
        loop = RetryLoop(generator=gen, max_retries=3)
        result = loop.solve("What is some number?")

        assert result.num_attempts == 1
        assert result.final_answer == "99"
        # correct is False because we can't verify without expected
        assert result.correct is False  # answer_correct is None -> not True

    def test_solve_result_contains_all_attempts(self):
        """SolveResult should record every attempt."""
        codes = [
            "answer = 1 / 0\n",
            "answer = wrong_var\n",
            'answer = 5\nprint(f"Answer: {answer}")\n',
        ]
        gen = _make_generator_with_sequence(codes)
        loop = RetryLoop(generator=gen, max_retries=3)
        result = loop.solve("What is 2 + 3?", expected_answer="5")

        assert len(result.attempts) == 3
        assert result.attempts[0].attempt_number == 1
        assert result.attempts[1].attempt_number == 2
        assert result.attempts[2].attempt_number == 3
