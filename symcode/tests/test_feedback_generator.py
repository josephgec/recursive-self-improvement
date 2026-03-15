"""Tests for the feedback generator."""

from __future__ import annotations

import pytest

from src.verification.feedback_generator import FeedbackGenerator
from src.verification.result_types import CodeError, CodeExecutionResult


class TestFeedbackGenerate:
    """Test feedback generation for different error types."""

    def setup_method(self):
        self.gen = FeedbackGenerator()

    def test_success_feedback(self):
        result = CodeExecutionResult(
            success=True, answer="42", stdout="Answer: 42"
        )
        feedback = self.gen.generate(result)
        assert "successfully" in feedback.lower()
        assert "42" in feedback

    def test_no_error_details(self):
        result = CodeExecutionResult(success=False, error=None)
        feedback = self.gen.generate(result)
        assert "no error details" in feedback.lower()

    def test_syntax_error_feedback(self):
        error = CodeError(
            error_type="SyntaxError",
            message="invalid syntax",
            line_number=5,
        )
        result = CodeExecutionResult(success=False, error=error)
        code = "line1\nline2\nline3\nline4\nline5_has_error\nline6\nline7"
        feedback = self.gen.generate(result, code=code)
        assert "SyntaxError" in feedback
        assert "syntax_basic" in feedback.lower()
        assert "LINE: 5" in feedback
        assert "CODE CONTEXT" in feedback
        assert "line5_has_error" in feedback
        assert "FIX HINT" in feedback

    def test_name_error_feedback(self):
        error = CodeError(
            error_type="NameError",
            message="name 'x' is not defined",
        )
        result = CodeExecutionResult(success=False, error=error)
        feedback = self.gen.generate(result)
        assert "NameError" in feedback
        assert "name 'x' is not defined" in feedback
        assert "FIX HINT" in feedback

    def test_timeout_feedback(self):
        error = CodeError(
            error_type="TimeoutError",
            message="Execution timed out after 30s",
        )
        result = CodeExecutionResult(
            success=False, error=error, timed_out=True
        )
        feedback = self.gen.generate(result)
        assert "TimeoutError" in feedback
        assert "time limit" in feedback.lower()

    def test_stderr_included(self):
        error = CodeError(error_type="ValueError", message="bad value")
        result = CodeExecutionResult(
            success=False, error=error, stderr="Traceback: ...\nValueError: bad value"
        )
        feedback = self.gen.generate(result)
        assert "STDERR" in feedback
        assert "Traceback" in feedback

    def test_stderr_truncated(self):
        error = CodeError(error_type="ValueError", message="bad value")
        long_stderr = "x" * 600
        result = CodeExecutionResult(
            success=False, error=error, stderr=long_stderr
        )
        feedback = self.gen.generate(result)
        assert "truncated" in feedback.lower()

    def test_feedback_truncated_to_budget(self):
        error = CodeError(
            error_type="RuntimeError",
            message="A" * 3000,
        )
        result = CodeExecutionResult(
            success=False, error=error, stderr="B" * 3000
        )
        feedback = self.gen.generate(result, code="C\n" * 1000)
        assert len(feedback) <= 2100  # budget + truncation message

    def test_line_context_with_code(self):
        """Line context should show surrounding lines with markers."""
        error = CodeError(
            error_type="NameError",
            message="name 'y' is not defined",
            line_number=3,
        )
        code = "import sympy\nx = 1\ny = undefined_var\nanswer = y"
        result = CodeExecutionResult(success=False, error=error)
        feedback = self.gen.generate(result, code=code)
        assert "CODE CONTEXT" in feedback
        assert " >> " in feedback  # marker on error line

    def test_zero_division_feedback(self):
        error = CodeError(
            error_type="ZeroDivisionError",
            message="division by zero",
        )
        result = CodeExecutionResult(success=False, error=error)
        feedback = self.gen.generate(result)
        assert "ZeroDivisionError" in feedback
        assert "division_zero" in feedback.lower()


class TestWrongAnswerFeedback:
    def setup_method(self):
        self.gen = FeedbackGenerator()

    def test_wrong_answer_with_expected(self):
        feedback = self.gen.generate_wrong_answer_feedback(
            got="4", expected="5", code="answer = 2 + 2"
        )
        assert "Wrong Answer" in feedback
        assert "YOUR ANSWER: 4" in feedback
        assert "EXPECTED: 5" in feedback
        assert "FIX HINT" in feedback

    def test_wrong_answer_without_expected(self):
        feedback = self.gen.generate_wrong_answer_feedback(got="4")
        assert "Wrong Answer" in feedback
        assert "YOUR ANSWER: 4" in feedback
        assert "EXPECTED" not in feedback

    def test_wrong_answer_truncation(self):
        """Very long feedback should be truncated."""
        long_answer = "x" * 3000
        feedback = self.gen.generate_wrong_answer_feedback(
            got=long_answer, expected="42"
        )
        assert len(feedback) <= 2100


class TestSuccessFeedback:
    def setup_method(self):
        self.gen = FeedbackGenerator()

    def test_success_feedback(self):
        result = CodeExecutionResult(success=True, answer="42")
        feedback = self.gen.generate_success_feedback(result)
        assert "successfully" in feedback.lower()
        assert "42" in feedback

    def test_success_feedback_none_answer(self):
        result = CodeExecutionResult(success=True, answer=None)
        feedback = self.gen.generate_success_feedback(result)
        assert "successfully" in feedback.lower()
