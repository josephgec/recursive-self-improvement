"""Financial mathematics benchmark tasks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FinancialTask:
    """A single financial math task."""

    task_id: str
    category: str
    question: str
    expected_answer: str
    difficulty: str = "medium"
    parameters: Dict = field(default_factory=dict)


class FinancialMathBenchmark:
    """Generates and manages financial math benchmark tasks.

    Supports 7 categories with programmatically generated tasks
    and correct answers computed via formulas.
    """

    CATEGORIES = [
        "compound_interest",
        "present_value",
        "loan_amortization",
        "option_pricing",
        "risk_return",
        "bond_valuation",
        "time_value",
    ]

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._tasks: List[FinancialTask] = []
        self._generated = False

    def generate_tasks(self, n_per_category: int = 8) -> List[FinancialTask]:
        """Generate benchmark tasks across all categories.

        Args:
            n_per_category: Number of tasks per category

        Returns:
            List of FinancialTask objects with correct answers.
        """
        self._tasks = []

        generators = {
            "compound_interest": self._gen_compound_interest,
            "present_value": self._gen_present_value,
            "loan_amortization": self._gen_loan_amortization,
            "option_pricing": self._gen_option_pricing,
            "risk_return": self._gen_risk_return,
            "bond_valuation": self._gen_bond_valuation,
            "time_value": self._gen_time_value,
        }

        for category, gen_fn in generators.items():
            for i in range(n_per_category):
                task = gen_fn(i)
                self._tasks.append(task)

        self._generated = True
        return list(self._tasks)

    def load(self) -> List[FinancialTask]:
        """Load or generate tasks."""
        if not self._generated:
            self.generate_tasks()
        return list(self._tasks)

    def get_tasks_by_category(self, category: str) -> List[FinancialTask]:
        """Get tasks for a specific category."""
        if not self._generated:
            self.generate_tasks()
        return [t for t in self._tasks if t.category == category]

    def _gen_compound_interest(self, idx: int) -> FinancialTask:
        """Generate compound interest problem.

        Formula: A = P * (1 + r/n)^(n*t)
        """
        principal = self._rng.choice([1000, 5000, 10000, 25000, 50000])
        rate = self._rng.choice([0.03, 0.05, 0.06, 0.08, 0.10, 0.12])
        years = self._rng.choice([1, 2, 3, 5, 10])
        compounds = self._rng.choice([1, 2, 4, 12])

        compound_names = {1: "annually", 2: "semi-annually", 4: "quarterly", 12: "monthly"}
        compound_name = compound_names[compounds]

        amount = principal * (1 + rate / compounds) ** (compounds * years)
        answer = round(amount, 2)

        question = (
            f"Calculate the future value of ${principal:,.2f} invested at "
            f"{rate*100:.1f}% annual interest rate, compounded {compound_name}, "
            f"after {years} year{'s' if years > 1 else ''}."
        )

        return FinancialTask(
            task_id=f"ci_{idx}",
            category="compound_interest",
            question=question,
            expected_answer=str(answer),
            parameters={
                "principal": principal,
                "rate": rate,
                "years": years,
                "compounds_per_year": compounds,
            },
        )

    def _gen_present_value(self, idx: int) -> FinancialTask:
        """Generate present value problem.

        Formula: PV = FV / (1 + r)^t
        """
        future_value = self._rng.choice([5000, 10000, 25000, 50000, 100000])
        rate = self._rng.choice([0.03, 0.05, 0.07, 0.08, 0.10])
        years = self._rng.choice([1, 3, 5, 10, 15, 20])

        pv = future_value / (1 + rate) ** years
        answer = round(pv, 2)

        question = (
            f"What is the present value of ${future_value:,.2f} to be received "
            f"in {years} years, assuming a discount rate of {rate*100:.1f}%?"
        )

        return FinancialTask(
            task_id=f"pv_{idx}",
            category="present_value",
            question=question,
            expected_answer=str(answer),
            parameters={
                "future_value": future_value,
                "rate": rate,
                "years": years,
            },
        )

    def _gen_loan_amortization(self, idx: int) -> FinancialTask:
        """Generate loan amortization problem.

        Formula: PMT = P * [r(1+r)^n] / [(1+r)^n - 1]
        """
        principal = self._rng.choice([10000, 50000, 100000, 200000, 500000])
        annual_rate = self._rng.choice([0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        years = self._rng.choice([5, 10, 15, 20, 30])

        monthly_rate = annual_rate / 12
        n_payments = years * 12

        if monthly_rate > 0:
            payment = principal * (
                monthly_rate * (1 + monthly_rate) ** n_payments
            ) / ((1 + monthly_rate) ** n_payments - 1)
        else:
            payment = principal / n_payments

        answer = round(payment, 2)

        question = (
            f"Calculate the monthly payment for a ${principal:,.2f} loan "
            f"at {annual_rate*100:.1f}% annual interest rate over {years} years."
        )

        return FinancialTask(
            task_id=f"la_{idx}",
            category="loan_amortization",
            question=question,
            expected_answer=str(answer),
            parameters={
                "principal": principal,
                "annual_rate": annual_rate,
                "years": years,
            },
        )

    def _gen_option_pricing(self, idx: int) -> FinancialTask:
        """Generate simplified option pricing problem.

        Uses simplified Black-Scholes approximation for educational purposes.
        Intrinsic value: max(S - K, 0) for calls, max(K - S, 0) for puts.
        """
        stock_price = self._rng.choice([50, 75, 100, 120, 150, 200])
        strike_price = stock_price + self._rng.choice([-20, -10, -5, 0, 5, 10, 20])
        is_call = self._rng.choice([True, False])

        if is_call:
            intrinsic = max(stock_price - strike_price, 0)
            option_type = "call"
        else:
            intrinsic = max(strike_price - stock_price, 0)
            option_type = "put"

        answer = round(intrinsic, 2)

        question = (
            f"What is the intrinsic value of a {option_type} option with a "
            f"strike price of ${strike_price} when the stock is trading at ${stock_price}?"
        )

        return FinancialTask(
            task_id=f"op_{idx}",
            category="option_pricing",
            question=question,
            expected_answer=str(answer),
            parameters={
                "stock_price": stock_price,
                "strike_price": strike_price,
                "option_type": option_type,
            },
        )

    def _gen_risk_return(self, idx: int) -> FinancialTask:
        """Generate risk-return problem.

        Expected return: E(R) = sum(prob_i * return_i)
        """
        scenarios = self._rng.choice([
            [("boom", 0.3, 0.20), ("normal", 0.5, 0.10), ("recession", 0.2, -0.05)],
            [("high", 0.25, 0.25), ("medium", 0.50, 0.12), ("low", 0.25, 0.02)],
            [("best", 0.2, 0.30), ("good", 0.3, 0.15), ("fair", 0.3, 0.05), ("poor", 0.2, -0.10)],
        ])

        expected_return = sum(prob * ret for _, prob, ret in scenarios)
        answer = round(expected_return * 100, 2)

        scenario_text = ", ".join(
            f"{name}: {prob*100:.0f}% probability with {ret*100:.1f}% return"
            for name, prob, ret in scenarios
        )

        question = (
            f"Calculate the expected return (in %) given these scenarios: {scenario_text}."
        )

        return FinancialTask(
            task_id=f"rr_{idx}",
            category="risk_return",
            question=question,
            expected_answer=str(answer),
            parameters={"scenarios": scenarios},
        )

    def _gen_bond_valuation(self, idx: int) -> FinancialTask:
        """Generate bond valuation problem.

        Bond price = sum of PV of coupons + PV of face value.
        Price = C * [1 - (1+r)^-n] / r + F / (1+r)^n
        """
        face_value = self._rng.choice([1000, 5000, 10000])
        coupon_rate = self._rng.choice([0.04, 0.05, 0.06, 0.07, 0.08])
        yield_rate = self._rng.choice([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
        years = self._rng.choice([5, 10, 15, 20])

        coupon = face_value * coupon_rate

        if yield_rate > 0:
            pv_coupons = coupon * (1 - (1 + yield_rate) ** (-years)) / yield_rate
        else:
            pv_coupons = coupon * years

        pv_face = face_value / (1 + yield_rate) ** years
        price = pv_coupons + pv_face
        answer = round(price, 2)

        question = (
            f"Calculate the price of a bond with ${face_value:,.2f} face value, "
            f"{coupon_rate*100:.1f}% coupon rate, {years} years to maturity, "
            f"and a yield to maturity of {yield_rate*100:.1f}%."
        )

        return FinancialTask(
            task_id=f"bv_{idx}",
            category="bond_valuation",
            question=question,
            expected_answer=str(answer),
            parameters={
                "face_value": face_value,
                "coupon_rate": coupon_rate,
                "yield_rate": yield_rate,
                "years": years,
            },
        )

    def _gen_time_value(self, idx: int) -> FinancialTask:
        """Generate time value of money problem.

        Future value of annuity: FV = PMT * [(1+r)^n - 1] / r
        """
        payment = self._rng.choice([100, 250, 500, 1000, 2000])
        rate = self._rng.choice([0.03, 0.05, 0.06, 0.08, 0.10])
        years = self._rng.choice([5, 10, 15, 20, 25, 30])

        if rate > 0:
            fv = payment * ((1 + rate) ** years - 1) / rate
        else:
            fv = payment * years

        answer = round(fv, 2)

        question = (
            f"Calculate the future value of an annuity with annual payments "
            f"of ${payment:,.2f}, an interest rate of {rate*100:.1f}%, "
            f"over {years} years."
        )

        return FinancialTask(
            task_id=f"tv_{idx}",
            category="time_value",
            question=question,
            expected_answer=str(answer),
            parameters={
                "payment": payment,
                "rate": rate,
                "years": years,
            },
        )

    def to_eval_tasks(
        self, tasks: Optional[List[FinancialTask]] = None
    ) -> List[Dict]:
        """Convert FinancialTasks to eval task dicts."""
        if tasks is None:
            tasks = self.load()
        return [
            {
                "task_id": t.task_id,
                "question": t.question,
                "expected_answer": t.expected_answer,
                "category": t.category,
            }
            for t in tasks
        ]
