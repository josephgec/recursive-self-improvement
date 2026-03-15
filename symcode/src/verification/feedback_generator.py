"""Generate structured feedback for the LLM retry loop."""

from __future__ import annotations

from typing import Any

from src.verification.error_classifier import ErrorCategory, ErrorClassifier
from src.verification.result_types import CodeError, CodeExecutionResult
from src.utils.logging import get_logger

logger = get_logger("feedback_generator")

# Target: keep feedback under this many tokens (approx 4 chars per token)
_MAX_FEEDBACK_CHARS = 2000  # ~500 tokens


class FeedbackGenerator:
    """Generate structured error feedback for the retry loop."""

    def __init__(self):
        self.classifier = ErrorClassifier()

    def generate(
        self,
        execution_result: CodeExecutionResult,
        code: str = "",
    ) -> str:
        """Generate feedback for a failed execution.

        Returns a structured string under ~500 tokens that the LLM
        can use to correct its code.
        """
        if execution_result.success:
            return self.generate_success_feedback(execution_result)

        error = execution_result.error
        if error is None:
            return "Execution failed with no error details."

        category, hint = self.classifier.classify_and_hint(error)

        parts: list[str] = []
        parts.append(f"ERROR TYPE: {error.error_type}")
        parts.append(f"CATEGORY: {category.value}")
        parts.append(f"MESSAGE: {error.message}")

        if error.line_number is not None:
            parts.append(f"LINE: {error.line_number}")
            # Show context around the error line
            if code:
                lines = code.split("\n")
                start = max(0, error.line_number - 3)
                end = min(len(lines), error.line_number + 2)
                context_lines = []
                for i in range(start, end):
                    marker = " >> " if i == error.line_number - 1 else "    "
                    context_lines.append(f"{marker}{i + 1}: {lines[i]}")
                parts.append("CODE CONTEXT:\n" + "\n".join(context_lines))

        if execution_result.timed_out:
            parts.append(
                "The code exceeded the time limit. "
                "Use a more efficient algorithm or SymPy built-in functions."
            )

        parts.append(f"FIX HINT: {hint}")

        # Truncate stderr if present
        if execution_result.stderr:
            stderr_excerpt = execution_result.stderr[:500]
            if len(execution_result.stderr) > 500:
                stderr_excerpt += "\n... (truncated)"
            parts.append(f"STDERR:\n{stderr_excerpt}")

        feedback = "\n".join(parts)

        # Truncate to budget
        if len(feedback) > _MAX_FEEDBACK_CHARS:
            feedback = feedback[:_MAX_FEEDBACK_CHARS] + "\n... (truncated)"

        return feedback

    def generate_wrong_answer_feedback(
        self,
        got: str,
        expected: str | None = None,
        code: str = "",
    ) -> str:
        """Generate feedback when code ran but produced wrong answer."""
        parts: list[str] = []
        parts.append("ERROR TYPE: Wrong Answer")
        parts.append("CATEGORY: logic_error")
        parts.append(f"YOUR ANSWER: {got}")
        if expected:
            parts.append(f"EXPECTED: {expected}")
        parts.append(
            "FIX HINT: The code executed successfully but produced "
            "the wrong answer. Review your mathematical approach. "
            "Check for off-by-one errors, incorrect formulas, or "
            "misinterpretation of the problem."
        )

        feedback = "\n".join(parts)
        if len(feedback) > _MAX_FEEDBACK_CHARS:
            feedback = feedback[:_MAX_FEEDBACK_CHARS] + "\n... (truncated)"
        return feedback

    def generate_success_feedback(
        self,
        execution_result: CodeExecutionResult,
    ) -> str:
        """Generate feedback for successful execution (informational)."""
        answer = execution_result.answer
        return f"Code executed successfully. Answer: {answer}"
