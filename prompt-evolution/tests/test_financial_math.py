"""Tests for financial math benchmark and answer checking."""

import math

import pytest

from src.evaluation.financial_math import FinancialMathBenchmark, FinancialTask
from src.evaluation.answer_checker import FinancialAnswerChecker
from src.evaluation.task_generator import generate_task_batch, generate_mixed_batch
from src.evaluation.general_bench import GeneralBenchmark


class TestFinancialTask:
    """Tests for FinancialTask dataclass."""

    def test_creation(self):
        task = FinancialTask(
            task_id="test_1",
            category="compound_interest",
            question="Calculate CI on $1000 at 5% for 2 years",
            expected_answer="1102.50",
        )
        assert task.task_id == "test_1"
        assert task.category == "compound_interest"

    def test_difficulty_default(self):
        task = FinancialTask(
            task_id="t", category="c", question="q", expected_answer="a"
        )
        assert task.difficulty == "medium"


class TestFinancialMathBenchmark:
    """Tests for task generation and formula verification."""

    def test_generate_tasks(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=3)
        assert len(tasks) == 3 * 7  # 7 categories

    def test_all_categories_represented(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=2)
        categories = {t.category for t in tasks}
        for cat in FinancialMathBenchmark.CATEGORIES:
            assert cat in categories

    def test_compound_interest_formula(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=5)
        ci_tasks = [t for t in tasks if t.category == "compound_interest"]
        assert len(ci_tasks) == 5

        for task in ci_tasks:
            params = task.parameters
            p = params["principal"]
            r = params["rate"]
            n = params["compounds_per_year"]
            t_years = params["years"]
            expected = p * (1 + r / n) ** (n * t_years)
            assert float(task.expected_answer) == pytest.approx(expected, rel=0.001)

    def test_present_value_formula(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=3)
        pv_tasks = [t for t in tasks if t.category == "present_value"]

        for task in pv_tasks:
            params = task.parameters
            fv = params["future_value"]
            r = params["rate"]
            t_years = params["years"]
            expected = fv / (1 + r) ** t_years
            assert float(task.expected_answer) == pytest.approx(expected, rel=0.001)

    def test_loan_amortization_formula(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=3)
        la_tasks = [t for t in tasks if t.category == "loan_amortization"]

        for task in la_tasks:
            params = task.parameters
            p = params["principal"]
            annual_r = params["annual_rate"]
            years = params["years"]
            monthly_r = annual_r / 12
            n = years * 12
            expected = p * (monthly_r * (1 + monthly_r) ** n) / (
                (1 + monthly_r) ** n - 1
            )
            assert float(task.expected_answer) == pytest.approx(expected, rel=0.001)

    def test_option_pricing_intrinsic(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=3)
        op_tasks = [t for t in tasks if t.category == "option_pricing"]

        for task in op_tasks:
            params = task.parameters
            s = params["stock_price"]
            k = params["strike_price"]
            if params["option_type"] == "call":
                expected = max(s - k, 0)
            else:
                expected = max(k - s, 0)
            assert float(task.expected_answer) == pytest.approx(expected, rel=0.001)

    def test_risk_return_expected(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=3)
        rr_tasks = [t for t in tasks if t.category == "risk_return"]

        for task in rr_tasks:
            scenarios = task.parameters["scenarios"]
            expected = sum(prob * ret for _, prob, ret in scenarios) * 100
            assert float(task.expected_answer) == pytest.approx(expected, rel=0.01)

    def test_bond_valuation_formula(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=3)
        bv_tasks = [t for t in tasks if t.category == "bond_valuation"]

        for task in bv_tasks:
            params = task.parameters
            fv = params["face_value"]
            cr = params["coupon_rate"]
            yr = params["yield_rate"]
            years = params["years"]
            coupon = fv * cr
            pv_coupons = coupon * (1 - (1 + yr) ** (-years)) / yr
            pv_face = fv / (1 + yr) ** years
            expected = pv_coupons + pv_face
            assert float(task.expected_answer) == pytest.approx(expected, rel=0.001)

    def test_time_value_annuity(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=3)
        tv_tasks = [t for t in tasks if t.category == "time_value"]

        for task in tv_tasks:
            params = task.parameters
            pmt = params["payment"]
            r = params["rate"]
            years = params["years"]
            expected = pmt * ((1 + r) ** years - 1) / r
            assert float(task.expected_answer) == pytest.approx(expected, rel=0.001)

    def test_load_generates_if_needed(self):
        bench = FinancialMathBenchmark(seed=123)
        tasks = bench.load()
        assert len(tasks) > 0

    def test_get_tasks_by_category(self, financial_benchmark):
        financial_benchmark.generate_tasks(n_per_category=3)
        ci = financial_benchmark.get_tasks_by_category("compound_interest")
        assert len(ci) == 3
        assert all(t.category == "compound_interest" for t in ci)

    def test_to_eval_tasks(self, financial_benchmark):
        tasks = financial_benchmark.generate_tasks(n_per_category=2)
        eval_tasks = financial_benchmark.to_eval_tasks(tasks[:5])
        assert len(eval_tasks) == 5
        assert "task_id" in eval_tasks[0]
        assert "question" in eval_tasks[0]
        assert "expected_answer" in eval_tasks[0]

    def test_50_plus_tasks(self):
        bench = FinancialMathBenchmark(seed=42)
        tasks = bench.generate_tasks(n_per_category=8)
        assert len(tasks) >= 50

    def test_reproducibility(self):
        bench1 = FinancialMathBenchmark(seed=42)
        bench2 = FinancialMathBenchmark(seed=42)
        tasks1 = bench1.generate_tasks(n_per_category=3)
        tasks2 = bench2.generate_tasks(n_per_category=3)
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.expected_answer == t2.expected_answer


class TestFinancialAnswerChecker:
    """Tests for answer checking."""

    def test_exact_match(self):
        checker = FinancialAnswerChecker()
        assert checker.check("The answer is 1102.50", "1102.50") is True

    def test_with_dollar_sign(self):
        checker = FinancialAnswerChecker()
        assert checker.check("The result is $1,234.56", "1234.56") is True

    def test_with_commas(self):
        checker = FinancialAnswerChecker()
        assert checker.check("Value: 10,000.00", "10000.00") is True

    def test_percentage(self):
        checker = FinancialAnswerChecker()
        assert checker.check("Return is 8.5%", "8.5") is True

    def test_within_tolerance(self):
        checker = FinancialAnswerChecker(tolerance=0.02)
        assert checker.check("The answer is 1100", "1102.50") is True  # Within 2%

    def test_outside_tolerance(self):
        checker = FinancialAnswerChecker(tolerance=0.01)
        assert checker.check("The answer is 1000", "1102.50") is False

    def test_extract_numeric_dollar(self):
        checker = FinancialAnswerChecker()
        assert checker._extract_numeric("$1,234.56") == pytest.approx(1234.56)

    def test_extract_numeric_percent(self):
        checker = FinancialAnswerChecker()
        assert checker._extract_numeric("8.5%") == pytest.approx(8.5)

    def test_extract_numeric_plain(self):
        checker = FinancialAnswerChecker()
        assert checker._extract_numeric("42.5") == pytest.approx(42.5)

    def test_extract_numeric_negative(self):
        checker = FinancialAnswerChecker()
        assert checker._extract_numeric("-5.00") == pytest.approx(-5.0)

    def test_extract_numeric_invalid(self):
        checker = FinancialAnswerChecker()
        assert checker._extract_numeric("no numbers here") is None

    def test_extract_numeric_empty(self):
        checker = FinancialAnswerChecker()
        assert checker._extract_numeric("") is None

    def test_compare_numeric_zero_expected(self):
        checker = FinancialAnswerChecker(tolerance=0.02)
        assert checker._compare_numeric(0.0, 0.0) is True
        assert checker._compare_numeric(0.001, 0.0) is True
        assert checker._compare_numeric(1.0, 0.0) is False

    def test_string_fallback(self):
        checker = FinancialAnswerChecker()
        assert checker.check("The answer is yes", "yes") is True

    def test_extract_all_numerics(self):
        checker = FinancialAnswerChecker()
        nums = checker._extract_all_numerics("Values are $100, $200, and $300.50")
        assert 100.0 in nums
        assert 200.0 in nums
        assert 300.50 in nums

    def test_check_multiple_numbers_in_response(self):
        checker = FinancialAnswerChecker(tolerance=0.01)
        response = "Step 1: P = 1000, r = 0.05, t = 2. Final answer: 1102.50"
        assert checker.check(response, "1102.50") is True


class TestTaskGenerator:
    """Tests for task generation utilities."""

    def test_generate_task_batch(self):
        tasks = generate_task_batch("compound_interest", n=5, seed=42)
        assert len(tasks) == 5
        assert all(t["category"] == "compound_interest" for t in tasks)

    def test_generate_mixed_batch(self):
        tasks = generate_mixed_batch(n=14, seed=42)
        assert len(tasks) == 14

    def test_generate_task_batch_with_seed(self):
        tasks1 = generate_task_batch("present_value", n=3, seed=42)
        tasks2 = generate_task_batch("present_value", n=3, seed=42)
        assert tasks1[0]["expected_answer"] == tasks2[0]["expected_answer"]


class TestGeneralBenchmark:
    """Tests for general benchmark placeholder."""

    def test_generate_tasks(self):
        bench = GeneralBenchmark(domain="test")
        tasks = bench.generate_tasks(n=5)
        assert len(tasks) == 5
        assert tasks[0].domain == "test"

    def test_to_eval_tasks(self):
        bench = GeneralBenchmark()
        bench.generate_tasks(n=3)
        eval_tasks = bench.to_eval_tasks()
        assert len(eval_tasks) == 3
        assert "question" in eval_tasks[0]
