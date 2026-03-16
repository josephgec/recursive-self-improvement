"""ARC-AGI benchmark with 15+ built-in pattern/transformation tasks."""

from __future__ import annotations

from src.benchmarks.registry import BaseBenchmark, BenchmarkTask
from src.benchmarks.answer_checking import ExactChecker


class ARCAGIBenchmark(BaseBenchmark):
    name = "arc_agi"

    def _build_tasks(self) -> None:
        self._tasks = []

        # Color swap tasks (5)
        color_swap = [
            ("cs_01", "Swap colors: [[1,0],[0,1]] -> swap 0 and 1", [[0, 1], [1, 0]]),
            ("cs_02", "Swap colors: [[2,0,2],[0,2,0]] -> swap 0 and 2", [[0, 2, 0], [2, 0, 2]]),
            ("cs_03", "Swap colors: [[1,1],[0,0]] -> swap 0 and 1", [[0, 0], [1, 1]]),
            ("cs_04", "Swap colors: [[3,0],[0,3],[3,0]] -> swap 0 and 3",
             [[0, 3], [3, 0], [0, 3]]),
            ("cs_05", "Swap colors: [[1,2],[2,1]] -> swap 1 and 2", [[2, 1], [1, 2]]),
        ]
        for tid, prompt, ans in color_swap:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="color_swap",
                prompt=prompt, expected_answer=ans,
            ))

        # Pattern recognition tasks (5)
        pattern = [
            ("pat_01", "Complete: [1,2,3,?] arithmetic +1", [1, 2, 3, 4]),
            ("pat_02", "Complete: [2,4,8,?] geometric *2", [2, 4, 8, 16]),
            ("pat_03", "Mirror: [[1,2],[3,4]] horizontal flip", [[2, 1], [4, 3]]),
            ("pat_04", "Repeat: [[1,0]] -> 3 times vertically",
             [[1, 0], [1, 0], [1, 0]]),
            ("pat_05", "Fill: [[1,0,1],[0,0,0],[1,0,1]] center with 1",
             [[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
        ]
        for tid, prompt, ans in pattern:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="pattern",
                prompt=prompt, expected_answer=ans,
            ))

        # Transform tasks (6)
        transform = [
            ("tr_01", "Rotate 90 CW: [[1,2],[3,4]]", [[3, 1], [4, 2]]),
            ("tr_02", "Transpose: [[1,2,3],[4,5,6]]", [[1, 4], [2, 5], [3, 6]]),
            ("tr_03", "Vertical flip: [[1,2],[3,4]]", [[3, 4], [1, 2]]),
            ("tr_04", "Scale 2x: [[1]]", [[1, 1], [1, 1]]),
            ("tr_05", "Crop non-zero: [[0,1,0],[0,2,0],[0,0,0]]", [[1], [2]]),
            ("tr_06", "Invert: [[1,0],[0,1]] (0->1, 1->0)", [[0, 1], [1, 0]]),
        ]
        for tid, prompt, ans in transform:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="transform",
                prompt=prompt, expected_answer=ans,
            ))

    def check_answer(self, task, predicted):
        return ExactChecker.check(predicted, task.expected_answer)
