"""RLMCodeExecutor: execute LLM-generated code blocks in a sandboxed REPL."""

from __future__ import annotations

import io
import re
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.context_loader import ContextMeta
from src.core.result_protocol import ResultProtocol, FinalSignal
from src.strategies.peek_helpers import make_peek
from src.strategies.grep_helpers import make_grep
from src.strategies.chunk_helpers import make_chunk


@dataclass
class CodeBlockResult:
    """Result of executing a single code block."""
    code: str
    stdout: str
    stderr: str
    success: bool
    final_signal: Optional[FinalSignal] = None
    exception: Optional[str] = None


class RLMCodeExecutor:
    """Execute code blocks produced by the LLM inside a REPL namespace."""

    def __init__(
        self,
        max_iterations: int = 10,
        max_output_lines: int = 500,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_output_lines = max_output_lines
        self.repl: Dict[str, Any] = {}
        self.iteration = 0

    def setup(self, context_meta: ContextMeta) -> None:
        """Prepare the REPL namespace with helpers and protocol functions."""
        # Inject result protocol
        ResultProtocol.inject_protocol_functions(self.repl)

        # Inject helper functions
        self.repl["peek"] = make_peek(self.repl)
        grep_fn, search_fn = make_grep(self.repl)
        self.repl["grep"] = grep_fn
        self.repl["search"] = search_fn
        chunk_fn, count_lines_fn = make_chunk(self.repl)
        self.repl["chunk"] = chunk_fn
        self.repl["count_lines"] = count_lines_fn

        self.iteration = 0

    def execute_block(self, code: str) -> CodeBlockResult:
        """Execute a code block and return the result.

        Before execution, checks for FINAL signals.
        """
        self.iteration += 1

        # Check for FINAL signal
        signal = ResultProtocol.detect_final(code)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, self.repl)  # noqa: S102
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()

            # Truncate output
            stdout = self._truncate(stdout)

            return CodeBlockResult(
                code=code,
                stdout=stdout,
                stderr=stderr,
                success=True,
                final_signal=signal,
            )
        except Exception as exc:
            return CodeBlockResult(
                code=code,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                success=False,
                final_signal=signal,
                exception=str(exc),
            )

    def budget_remaining(self) -> int:
        """Return how many iterations remain."""
        return max(0, self.max_iterations - self.iteration)

    def budget_exhausted(self) -> bool:
        """Return True if no iterations remain."""
        return self.iteration >= self.max_iterations

    @staticmethod
    def extract_code_blocks(response: str) -> List[str]:
        """Extract Python code blocks from an LLM response.

        Looks for ```python ... ``` fenced blocks, or falls back to the
        entire response if no fences are found but it looks like code.
        """
        # Match ```python ... ``` blocks
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return [m.strip() for m in matches if m.strip()]

        # If no fenced blocks, check if the response looks like code
        lines = response.strip().split("\n")
        code_indicators = {"import ", "def ", "for ", "if ", "print(", "result", "=", "FINAL"}
        if any(any(ind in line for ind in code_indicators) for line in lines[:5]):
            return [response.strip()]

        return []

    def _truncate(self, text: str) -> str:
        """Truncate output to max_output_lines."""
        lines = text.split("\n")
        if len(lines) > self.max_output_lines:
            kept = lines[: self.max_output_lines]
            kept.append(f"... ({len(lines) - self.max_output_lines} lines truncated)")
            return "\n".join(kept)
        return text
