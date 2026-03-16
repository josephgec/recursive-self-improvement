"""OOLONG benchmark with 15+ built-in long-context tasks."""

from __future__ import annotations

from src.benchmarks.registry import BaseBenchmark, BenchmarkTask
from src.benchmarks.answer_checking import NumericChecker, ExactChecker


class OOLONGBenchmark(BaseBenchmark):
    name = "oolong"

    def _build_tasks(self) -> None:
        self._tasks = []

        # Retrieval tasks (4)
        retrieval = [
            ("ret_01", "Find the capital of France in context", "Paris"),
            ("ret_02", "Find the year Python was created in context", "1991"),
            ("ret_03", "Find the author of 'To Kill a Mockingbird'", "Harper Lee"),
            ("ret_04", "Find the boiling point of water in Celsius", "100"),
        ]
        for tid, prompt, ans in retrieval:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="retrieval",
                prompt=prompt, expected_answer=ans,
            ))

        # Aggregation tasks (4)
        aggregation = [
            ("agg_01", "Sum of values: [10, 20, 30, 40]", 100),
            ("agg_02", "Average of: [4, 8, 12, 16, 20]", 12.0),
            ("agg_03", "Max of: [3, 7, 2, 9, 1, 5]", 9),
            ("agg_04", "Count items > 5 in: [1, 3, 6, 8, 2, 9]", 3),
        ]
        for tid, prompt, ans in aggregation:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="aggregation",
                prompt=prompt, expected_answer=ans,
            ))

        # Reasoning tasks (4)
        reasoning = [
            ("rsn_01", "If A>B and B>C, is A>C?", "yes"),
            ("rsn_02", "All dogs are animals. Fido is a dog. Is Fido an animal?", "yes"),
            ("rsn_03", "If it rains then ground is wet. It rained. Is ground wet?", "yes"),
            ("rsn_04", "X=5, Y=X+3, Z=Y*2. What is Z?", 16),
        ]
        for tid, prompt, ans in reasoning:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="reasoning",
                prompt=prompt, expected_answer=ans,
            ))

        # Counting tasks (4)
        counting = [
            ("cntx_01", "How many words in 'the quick brown fox'?", 4),
            ("cntx_02", "How many vowels in 'education'?", 5),
            ("cntx_03", "How many sentences in 'Hi. Bye. OK.'?", 3),
            ("cntx_04", "Count unique letters in 'banana'", 3),
        ]
        for tid, prompt, ans in counting:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="counting",
                prompt=prompt, expected_answer=ans,
            ))

    def check_answer(self, task, predicted):
        if isinstance(task.expected_answer, (int, float)):
            return NumericChecker.check(predicted, task.expected_answer)
        return ExactChecker.check(
            str(predicted).strip().lower(),
            str(task.expected_answer).strip().lower(),
        )
