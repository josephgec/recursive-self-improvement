"""Data types for code execution and verification results."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeError:
    """Structured representation of a code execution error."""

    error_type: str  # e.g., "SyntaxError", "NameError"
    message: str
    line_number: int | None = None
    traceback: str = ""


@dataclass
class CodeExecutionResult:
    """Result of executing a code snippet."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    answer: Any = None
    error: CodeError | None = None
    execution_time: float = 0.0
    timed_out: bool = False


@dataclass
class AttemptRecord:
    """Record of a single solve attempt (generation + execution + check)."""

    attempt_number: int
    code: str
    execution_result: CodeExecutionResult
    extracted_answer: str | None = None
    answer_correct: bool | None = None
    feedback: str = ""
    generation_model: str = ""


@dataclass
class SolveResult:
    """Final result of the full solve pipeline (including retries)."""

    problem: str
    expected_answer: str | None = None
    final_answer: str | None = None
    correct: bool = False
    num_attempts: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    task_type: str = ""
    pipeline: str = "symcode"  # "symcode" or "prose"
    total_time: float = 0.0
