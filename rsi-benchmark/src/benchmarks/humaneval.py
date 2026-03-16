"""HumanEval benchmark with 15+ function completion tasks."""

from __future__ import annotations

from src.benchmarks.registry import BaseBenchmark, BenchmarkTask
from src.benchmarks.answer_checking import CodeChecker


class HumanEvalBenchmark(BaseBenchmark):
    name = "humaneval"

    def _build_tasks(self) -> None:
        self._tasks = []

        tasks = [
            ("he_01", "def add(a, b): return a + b", "add", [(2, 3, 5), (0, 0, 0), (-1, 1, 0)]),
            ("he_02", "def multiply(a, b): return a * b", "multiply",
             [(2, 3, 6), (0, 5, 0), (-2, 3, -6)]),
            ("he_03", "def max_val(lst): return max(lst)", "max_val",
             [([1, 3, 2], 3), ([5], 5), ([-1, -5], -1)]),
            ("he_04", "def min_val(lst): return min(lst)", "min_val",
             [([1, 3, 2], 1), ([5], 5), ([-1, -5], -5)]),
            ("he_05", "def is_even(n): return n % 2 == 0", "is_even",
             [(4, True), (3, False), (0, True)]),
            ("he_06", "def abs_val(n): return abs(n)", "abs_val",
             [(5, 5), (-3, 3), (0, 0)]),
            ("he_07", "def reverse_str(s): return s[::-1]", "reverse_str",
             [("hello", "olleh"), ("a", "a"), ("", "")]),
            ("he_08", "def len_list(lst): return len(lst)", "len_list",
             [([1, 2, 3], 3), ([], 0), ([1], 1)]),
            ("he_09", "def sum_list(lst): return sum(lst)", "sum_list",
             [([1, 2, 3], 6), ([], 0), ([10], 10)]),
            ("he_10", "def first_elem(lst): return lst[0]", "first_elem",
             [([1, 2, 3], 1), ([9], 9), (["a", "b"], "a")]),
            ("he_11", "def last_elem(lst): return lst[-1]", "last_elem",
             [([1, 2, 3], 3), ([9], 9), (["a", "b"], "b")]),
            ("he_12", "def square(n): return n * n", "square",
             [(3, 9), (0, 0), (-2, 4)]),
            ("he_13", "def to_upper(s): return s.upper()", "to_upper",
             [("hello", "HELLO"), ("Hi", "HI"), ("", "")]),
            ("he_14", "def concat(a, b): return a + b", "concat",
             [("hi", " there", "hi there"), ("", "x", "x")]),
            ("he_15", "def contains(lst, x): return x in lst", "contains",
             [([1, 2, 3], 2, True), ([1, 2], 5, False)]),
        ]

        for tid, code, func_name, test_cases in tasks:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="function_completion",
                prompt=f"Complete: {func_name}",
                expected_answer=code,
                metadata={"func_name": func_name, "test_cases": test_cases},
            ))

    def check_answer(self, task, predicted):
        return CodeChecker.check(predicted, task.expected_answer, task.metadata.get("test_cases", []))
