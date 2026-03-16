"""MATH500 benchmark with 30+ built-in tasks across mathematical domains."""

from __future__ import annotations

from src.benchmarks.registry import BaseBenchmark, BenchmarkTask
from src.benchmarks.answer_checking import NumericChecker


class MATH500Benchmark(BaseBenchmark):
    name = "math500"

    def _build_tasks(self) -> None:
        self._tasks = []

        # Algebra tasks (8)
        algebra = [
            ("alg_01", "Solve: 2x + 5 = 13", 4.0),
            ("alg_02", "Solve: x^2 - 9 = 0 (positive root)", 3.0),
            ("alg_03", "Simplify: (3x^2)(2x^3)", "6x^5"),
            ("alg_04", "Factor: x^2 + 5x + 6", "(x+2)(x+3)"),
            ("alg_05", "Solve: |2x - 1| = 7 (larger root)", 4.0),
            ("alg_06", "If f(x) = 3x + 2, find f(5)", 17.0),
            ("alg_07", "Sum of arithmetic series 1+2+...+100", 5050.0),
            ("alg_08", "Solve: 3(x-2) = 2(x+1)", 8.0),
        ]
        for tid, prompt, ans in algebra:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="algebra",
                prompt=prompt, expected_answer=ans,
            ))

        # Number theory tasks (7)
        number_theory = [
            ("nt_01", "GCD of 48 and 36", 12),
            ("nt_02", "LCM of 12 and 18", 36),
            ("nt_03", "Is 97 prime? (yes/no)", "yes"),
            ("nt_04", "Number of primes less than 20", 8),
            ("nt_05", "17 mod 5", 2),
            ("nt_06", "Euler's totient of 12", 4),
            ("nt_07", "Sum of divisors of 28", 56),
        ]
        for tid, prompt, ans in number_theory:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="number_theory",
                prompt=prompt, expected_answer=ans,
            ))

        # Calculus tasks (6)
        calculus = [
            ("calc_01", "Derivative of x^3 at x=2", 12.0),
            ("calc_02", "Integral of 2x from 0 to 3", 9.0),
            ("calc_03", "Limit of (x^2-1)/(x-1) as x->1", 2.0),
            ("calc_04", "Derivative of sin(x) at x=0", 1.0),
            ("calc_05", "Area under y=x^2 from 0 to 1", 1 / 3),
            ("calc_06", "d/dx(e^(2x)) at x=0", 2.0),
        ]
        for tid, prompt, ans in calculus:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="calculus",
                prompt=prompt, expected_answer=ans,
            ))

        # Geometry tasks (6)
        geometry = [
            ("geo_01", "Area of circle with radius 5", 78.53981633974483),
            ("geo_02", "Hypotenuse of 3-4-5 triangle", 5.0),
            ("geo_03", "Volume of cube with side 3", 27.0),
            ("geo_04", "Perimeter of rectangle 4x7", 22.0),
            ("geo_05", "Area of triangle base=6 height=8", 24.0),
            ("geo_06", "Diagonal of square with side 5", 7.0710678118654755),
        ]
        for tid, prompt, ans in geometry:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="geometry",
                prompt=prompt, expected_answer=ans,
            ))

        # Counting/combinatorics tasks (5)
        counting = [
            ("cnt_01", "5!", 120),
            ("cnt_02", "C(10,3)", 120),
            ("cnt_03", "P(5,2)", 20),
            ("cnt_04", "Ways to arrange letters in 'AAB'", 3),
            ("cnt_05", "Number of subsets of {1,2,3,4}", 16),
        ]
        for tid, prompt, ans in counting:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="counting",
                prompt=prompt, expected_answer=ans,
            ))

    def check_answer(self, task, predicted):
        if isinstance(task.expected_answer, (int, float)):
            return NumericChecker.check(predicted, task.expected_answer)
        return str(predicted).strip().lower() == str(task.expected_answer).strip().lower()
