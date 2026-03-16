"""SWE-Bench benchmark with 10+ software engineering tasks."""

from __future__ import annotations

from src.benchmarks.registry import BaseBenchmark, BenchmarkTask
from src.benchmarks.answer_checking import ExactChecker


class SWEBenchBenchmark(BaseBenchmark):
    name = "swebench"

    def _build_tasks(self) -> None:
        self._tasks = []

        # Bug fix tasks (6)
        bugfix = [
            ("bf_01", "Fix off-by-one: range(1, n) should be range(1, n+1)", "range(1, n+1)"),
            ("bf_02", "Fix: 'if x = 5' should be 'if x == 5'", "if x == 5"),
            ("bf_03", "Fix: missing return in function that computes sum", "return total"),
            ("bf_04", "Fix: list.append should not reassign list = list.append(x)",
             "list.append(x)"),
            ("bf_05", "Fix: string comparison 'is' should be '=='", "=="),
            ("bf_06", "Fix: division by zero - add check before dividing",
             "if denominator != 0"),
        ]
        for tid, prompt, ans in bugfix:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="bug_fix",
                prompt=prompt, expected_answer=ans,
            ))

        # Feature implementation tasks (6)
        feature = [
            ("ft_01", "Add a method to compute average of a list",
             "def average(lst): return sum(lst) / len(lst)"),
            ("ft_02", "Add input validation for positive integer",
             "if not isinstance(n, int) or n <= 0: raise ValueError"),
            ("ft_03", "Add logging to function entry",
             "logger.info('entering function')"),
            ("ft_04", "Add retry logic with max_retries parameter",
             "for attempt in range(max_retries)"),
            ("ft_05", "Add type hints to function signature",
             "def func(x: int, y: str) -> bool"),
            ("ft_06", "Add docstring to existing function",
             '"""Compute the result."""'),
        ]
        for tid, prompt, ans in feature:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="feature",
                prompt=prompt, expected_answer=ans,
            ))

    def check_answer(self, task, predicted):
        return ExactChecker.check(
            str(predicted).strip().lower(),
            str(task.expected_answer).strip().lower(),
        )
