"""Financial benchmark with 15+ finance/quantitative tasks."""

from __future__ import annotations

from src.benchmarks.registry import BaseBenchmark, BenchmarkTask
from src.benchmarks.answer_checking import NumericChecker, ExactChecker


class FinancialBenchmark(BaseBenchmark):
    name = "financial"

    def _build_tasks(self) -> None:
        self._tasks = []

        # Compound interest tasks (4)
        compound = [
            ("ci_01", "Compound interest: P=1000, r=5%, t=2y, annual", 1102.50),
            ("ci_02", "Compound interest: P=5000, r=8%, t=3y, annual", 6298.56),
            ("ci_03", "Compound interest: P=10000, r=3%, t=1y, annual", 10300.00),
            ("ci_04", "Simple interest: P=2000, r=4%, t=5y", 2400.00),
        ]
        for tid, prompt, ans in compound:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="compound_interest",
                prompt=prompt, expected_answer=ans,
            ))

        # Options pricing tasks (3)
        options = [
            ("opt_01", "Call option intrinsic value: S=105, K=100", 5.0),
            ("opt_02", "Put option intrinsic value: S=95, K=100", 5.0),
            ("opt_03", "Call option: S=90, K=100 (out of money)", 0.0),
        ]
        for tid, prompt, ans in options:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="options",
                prompt=prompt, expected_answer=ans,
            ))

        # Risk metrics tasks (4)
        risk = [
            ("rsk_01", "Sharpe ratio: return=12%, rf=2%, vol=15%", 0.6667),
            ("rsk_02", "Portfolio variance: w=[0.5,0.5], var=[0.04,0.09], corr=0", 0.0325),
            ("rsk_03", "Beta: cov(r_i, r_m)=0.03, var(r_m)=0.04", 0.75),
            ("rsk_04", "Expected return CAPM: rf=3%, beta=1.2, rm=10%", 11.4),
        ]
        for tid, prompt, ans in risk:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="risk",
                prompt=prompt, expected_answer=ans,
            ))

        # Bond tasks (4)
        bonds = [
            ("bnd_01", "Bond price: FV=1000, coupon=5%, YTM=5%, T=1", 1000.0),
            ("bnd_02", "Current yield: coupon=50, price=980", 5.102),
            ("bnd_03", "Zero coupon bond price: FV=1000, r=6%, T=5",
             747.26),
            ("bnd_04", "Yield to maturity approx: price=950, FV=1000, coupon=5%, T=10",
             5.66),
        ]
        for tid, prompt, ans in bonds:
            self._tasks.append(BenchmarkTask(
                task_id=tid, benchmark=self.name, category="bonds",
                prompt=prompt, expected_answer=ans,
            ))

    def check_answer(self, task, predicted):
        if isinstance(task.expected_answer, (int, float)):
            return NumericChecker.check(predicted, task.expected_answer, tolerance=0.02)
        return ExactChecker.check(
            str(predicted).strip().lower(),
            str(task.expected_answer).strip().lower(),
        )
