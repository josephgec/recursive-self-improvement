"""Sandboxed code execution via subprocess."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from src.verification.result_types import CodeError, CodeExecutionResult
from src.utils.logging import get_logger

logger = get_logger("executor")

# Wrapper script that executes the user code, captures the answer variable,
# and writes it as JSON to a results file.
_WRAPPER_TEMPLATE = textwrap.dedent("""\
    import json
    import sys
    import traceback

    _results_path = sys.argv[1]
    _results = {"success": False, "answer": None, "error": None}

    try:
        _namespace = {}
        exec(open(sys.argv[2]).read(), _namespace)
        if "answer" in _namespace:
            _ans = _namespace["answer"]
            try:
                _results["answer"] = str(_ans)
            except Exception:
                _results["answer"] = repr(_ans)
        _results["success"] = True
    except Exception as e:
        tb = traceback.format_exc()
        _results["error"] = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": tb,
        }
        # Try to extract line number
        import re as _re
        m = _re.search(r'File ".*?", line (\\d+)', tb)
        if m:
            _results["error"]["line_number"] = int(m.group(1))

    with open(_results_path, "w") as f:
        json.dump(_results, f)
""")


class SymCodeExecutor:
    """Execute Python code in a sandboxed subprocess."""

    def __init__(
        self,
        timeout: int = 30,
        python_executable: str | None = None,
    ):
        self.timeout = timeout
        self.python = python_executable or sys.executable

    def execute(self, code: str) -> CodeExecutionResult:
        """Execute code in a subprocess and return structured results.

        The code runs in a fresh Python process with access to sympy,
        numpy, fractions, etc.  A wrapper script captures the `answer`
        variable from the executed namespace.
        """
        start = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write wrapper and user code to temp files
            wrapper_path = tmpdir_path / "_wrapper.py"
            code_path = tmpdir_path / "_code.py"
            results_path = tmpdir_path / "_results.json"

            wrapper_path.write_text(_WRAPPER_TEMPLATE, encoding="utf-8")
            code_path.write_text(code, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [self.python, str(wrapper_path), str(results_path), str(code_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                return CodeExecutionResult(
                    success=False,
                    error=CodeError(
                        error_type="TimeoutError",
                        message=f"Execution timed out after {self.timeout}s",
                    ),
                    execution_time=elapsed,
                    timed_out=True,
                )

            elapsed = time.time() - start

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            # Read results JSON
            if results_path.exists():
                try:
                    results = json.loads(results_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    results = None
            else:
                results = None

            if results and results.get("success"):
                answer = results.get("answer")
                return CodeExecutionResult(
                    success=True,
                    stdout=stdout,
                    stderr=stderr,
                    answer=answer,
                    execution_time=elapsed,
                )

            # Error case
            error_data = (results or {}).get("error")
            if error_data:
                error = CodeError(
                    error_type=error_data.get("error_type", "UnknownError"),
                    message=error_data.get("message", ""),
                    line_number=error_data.get("line_number"),
                    traceback=error_data.get("traceback", ""),
                )
            elif stderr:
                # Parse error from stderr
                error = self._parse_stderr(stderr)
            else:
                error = CodeError(
                    error_type="UnknownError",
                    message="Execution failed with no output",
                )

            return CodeExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                error=error,
                execution_time=elapsed,
            )

    def _parse_stderr(self, stderr: str) -> CodeError:
        """Parse error information from stderr output."""
        # Try to find Python exception info
        lines = stderr.strip().split("\n")

        error_type = "UnknownError"
        message = stderr.strip()
        line_number = None

        # Look for "ErrorType: message" pattern
        for line in reversed(lines):
            m = re.match(r"^(\w+Error|\w+Exception):\s*(.+)$", line)
            if m:
                error_type = m.group(1)
                message = m.group(2)
                break

        # Look for line number
        for line in lines:
            m = re.search(r'File ".*?", line (\d+)', line)
            if m:
                line_number = int(m.group(1))

        return CodeError(
            error_type=error_type,
            message=message,
            line_number=line_number,
            traceback=stderr,
        )
