"""End-to-end integration test with mock LLM.

Scenarios:
1. First-attempt success
2. Self-correction (fail once, then succeed)
3. All-fail (exhaust retries)
4. Wrong-answer recovery (wrong answer, then correct)
5. Prose-only (SymCode fails, prose succeeds)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.benchmarks.math500 import BenchmarkProblem
from src.benchmarks.comparison import ComparisonAnalyzer
from src.benchmarks.metrics import MetricsComputer
from src.benchmarks.runner import BenchmarkResult, BenchmarkRunner
from src.analysis.error_taxonomy import ErrorTaxonomist
from src.analysis.retry_analysis import RetryAnalyzer
from src.analysis.report import generate_report
from src.pipeline.code_generator import SymCodeGenerator, GenerationResult
from src.pipeline.prose_baseline import ProseBaseline
from src.pipeline.router import TaskRouter
from src.verification.answer_checker import AnswerChecker
from src.verification.executor import SymCodeExecutor
from src.verification.result_types import (
    AttemptRecord,
    CodeError,
    CodeExecutionResult,
    SolveResult,
)
from src.verification.retry_loop import RetryLoop


# ── Test problems ───────────────────────────────────────────────────

PROBLEMS = [
    BenchmarkProblem(
        problem_id="test_001",
        problem="What is 2 + 3?",
        expected_answer="5",
        subject="algebra",
        difficulty=1,
    ),
    BenchmarkProblem(
        problem_id="test_002",
        problem="Solve for x: x^2 = 9, x > 0",
        expected_answer="3",
        subject="algebra",
        difficulty=2,
    ),
    BenchmarkProblem(
        problem_id="test_003",
        problem="Evaluate the integral of 1/x from 1 to infinity.",
        expected_answer="diverges",
        subject="calculus",
        difficulty=5,
    ),
    BenchmarkProblem(
        problem_id="test_004",
        problem="What is 10 choose 3?",
        expected_answer="120",
        subject="combinatorics",
        difficulty=2,
    ),
    BenchmarkProblem(
        problem_id="test_005",
        problem="Find the area of a circle with radius 3.",
        expected_answer="9*pi",
        subject="geometry",
        difficulty=1,
    ),
]


# ── Mock infrastructure ────────────────────────────────────────────


class MockExecutor:
    """Mock executor that returns pre-configured results per code snippet."""

    def __init__(self, result_map: dict[str, CodeExecutionResult]):
        """result_map: code_fragment -> CodeExecutionResult"""
        self._map = result_map
        self._call_count = 0

    def execute(self, code: str) -> CodeExecutionResult:
        self._call_count += 1
        for key, result in self._map.items():
            if key in code:
                return result
        # Default: success with answer 42
        return CodeExecutionResult(
            success=True,
            stdout="Answer: 42",
            answer="42",
            execution_time=0.1,
        )


def _make_gen_result(code: str, answer_value: str = "42") -> GenerationResult:
    """Helper to build a GenerationResult."""
    return GenerationResult(
        code=code,
        raw_response=f"```python\n{code}\n```",
        model="mock",
        prompt_tokens=50,
        completion_tokens=30,
    )


class ScenarioGenerator:
    """Mock SymCodeGenerator that returns different code per scenario.

    Tracks call count per problem to simulate retry behavior.
    """

    def __init__(self):
        self._call_counts: dict[str, int] = {}

    def _get_count(self, problem: str) -> int:
        count = self._call_counts.get(problem, 0) + 1
        self._call_counts[problem] = count
        return count

    def generate(
        self,
        problem: str,
        task_type: Any = None,
        use_cache: bool = True,
    ) -> GenerationResult:
        count = self._get_count(problem)
        return self._get_code(problem, count)

    def generate_correction(
        self,
        problem: str,
        prev_code: str,
        feedback: str,
        attempt: int = 1,
        max_attempts: int = 3,
        use_cache: bool = True,
    ) -> GenerationResult:
        count = self._get_count(problem)
        return self._get_code(problem, count)

    def _get_code(self, problem: str, attempt: int) -> GenerationResult:
        # Scenario 1: "2 + 3" -> always succeeds first try
        if "2 + 3" in problem:
            return _make_gen_result(
                "answer = 2 + 3\nprint(f'Answer: {answer}')",
                "5",
            )

        # Scenario 2: "x^2 = 9" -> fails first (syntax error), succeeds second
        if "x^2 = 9" in problem:
            if attempt == 1:
                return _make_gen_result(
                    "# Syntax error on purpose\nfor x in range(:\n  answer = x",
                )
            return _make_gen_result(
                "import sympy\nx = sympy.Symbol('x')\nsol = sympy.solve(x**2 - 9, x)\nanswer = max(sol)\nprint(f'Answer: {answer}')",
                "3",
            )

        # Scenario 3: "integral" -> always fails (generates bad code)
        if "integral" in problem.lower():
            return _make_gen_result(
                "# This will cause an error\nresult = undefined_func()\nanswer = result",
            )

        # Scenario 4: "choose" -> wrong answer first, correct second
        if "choose" in problem.lower():
            if attempt == 1:
                return _make_gen_result(
                    "answer = 100\nprint(f'Answer: {answer}')",
                    "100",
                )
            return _make_gen_result(
                "from math import comb\nanswer = comb(10, 3)\nprint(f'Answer: {answer}')",
                "120",
            )

        # Scenario 5: "area of a circle" -> always fails for SymCode
        if "area" in problem.lower() and "circle" in problem.lower():
            return _make_gen_result(
                "# Wrong approach\nanswer = 3 * 3\nprint(f'Answer: {answer}')",
                "9",
            )

        # Default
        return _make_gen_result("answer = 42\nprint(f'Answer: {answer}')")


class ScenarioProseBaseline:
    """Mock prose baseline with scenario-specific responses."""

    def __init__(self, **kwargs: Any):
        pass

    def solve(self, problem: str) -> tuple[str | None, str]:
        # Scenario 5: prose gets geometry right
        if "area" in problem.lower() and "circle" in problem.lower():
            return "9*pi", r"The area is $\pi r^2 = 9\pi$. \boxed{9\pi}"

        # Scenario 1: prose also gets simple problems right
        if "2 + 3" in problem:
            return "5", r"$2 + 3 = 5$. \boxed{5}"

        # Scenario 2: prose gets algebra wrong
        if "x^2 = 9" in problem:
            return "-3", r"$x = -3$. \boxed{-3}"

        # Scenario 3: prose fails too
        if "integral" in problem.lower():
            return "infinity", r"The integral is $\infty$."

        # Scenario 4: prose gets combinatorics wrong
        if "choose" in problem.lower():
            return "100", r"\boxed{100}"

        return "42", r"\boxed{42}"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_executor():
    """Mock executor with pre-configured results."""
    result_map = {
        # Scenario 1: success
        "answer = 2 + 3": CodeExecutionResult(
            success=True, stdout="Answer: 5", answer="5", execution_time=0.1,
        ),
        # Scenario 2: first attempt syntax error
        "for x in range(:": CodeExecutionResult(
            success=False,
            error=CodeError(
                error_type="SyntaxError",
                message="invalid syntax",
                line_number=2,
            ),
            execution_time=0.05,
        ),
        # Scenario 2: second attempt success
        "sympy.solve(x**2 - 9": CodeExecutionResult(
            success=True, stdout="Answer: 3", answer="3", execution_time=0.2,
        ),
        # Scenario 3: always fails (NameError)
        "undefined_func": CodeExecutionResult(
            success=False,
            error=CodeError(
                error_type="NameError",
                message="name 'undefined_func' is not defined",
                line_number=2,
            ),
            execution_time=0.05,
        ),
        # Scenario 4: first attempt wrong answer
        "answer = 100": CodeExecutionResult(
            success=True, stdout="Answer: 100", answer="100", execution_time=0.1,
        ),
        # Scenario 4: second attempt correct
        "comb(10, 3)": CodeExecutionResult(
            success=True, stdout="Answer: 120", answer="120", execution_time=0.1,
        ),
        # Scenario 5: wrong answer
        "answer = 3 * 3": CodeExecutionResult(
            success=True, stdout="Answer: 9", answer="9", execution_time=0.1,
        ),
    }
    return MockExecutor(result_map)


@pytest.fixture
def scenario_generator():
    return ScenarioGenerator()


@pytest.fixture
def integration_runner(scenario_generator, mock_executor):
    """Build a BenchmarkRunner with mock components."""
    checker = AnswerChecker()
    retry_loop = RetryLoop(
        generator=scenario_generator,
        executor=mock_executor,
        checker=checker,
        max_retries=3,
    )
    prose = ScenarioProseBaseline()

    runner = BenchmarkRunner(
        config={},
        generator=scenario_generator,
        retry_loop=retry_loop,
        prose_baseline=prose,
        checker=checker,
    )
    return runner


# ── Integration test ────────────────────────────────────────────────


class TestEndToEnd:
    """End-to-end integration test with 5 mock scenarios."""

    def test_full_pipeline(self, integration_runner, tmp_path):
        """Run all 5 scenarios and verify results."""
        result = integration_runner.run(
            PROBLEMS, pipelines=["symcode", "prose"], concurrency=1
        )

        # Should have results for all problems
        assert len(result.symcode_results) == 5
        assert len(result.prose_results) == 5

        # ── Scenario 1: First-attempt success ──────────────────────
        s1 = result.symcode_results[0]
        assert s1.correct is True
        assert s1.final_answer == "5"
        assert s1.num_attempts == 1

        # ── Scenario 2: Self-correction ────────────────────────────
        s2 = result.symcode_results[1]
        assert s2.correct is True
        assert s2.final_answer == "3"
        assert s2.num_attempts == 2
        # First attempt should have had an error
        assert s2.attempts[0].execution_result.error is not None
        assert s2.attempts[0].execution_result.error.error_type == "SyntaxError"

        # ── Scenario 3: All-fail ───────────────────────────────────
        s3 = result.symcode_results[2]
        assert s3.correct is False
        assert s3.num_attempts == 3  # exhausted retries

        # ── Scenario 4: Wrong-answer recovery ──────────────────────
        s4 = result.symcode_results[3]
        assert s4.correct is True
        assert s4.final_answer == "120"
        assert s4.num_attempts >= 2
        # First attempt had wrong answer
        assert s4.attempts[0].answer_correct is False

        # ── Scenario 5: Prose-only (SymCode fails, prose correct) ──
        s5 = result.symcode_results[4]
        assert s5.correct is False  # SymCode gets wrong answer (9 != 9*pi)
        p5 = result.prose_results[4]
        assert p5.correct is True  # Prose gets it right

        # ── Overall accuracy ───────────────────────────────────────
        result.compute_summaries()
        # SymCode: 3/5 (scenarios 1, 2, 4)
        assert result.symcode_accuracy == pytest.approx(3 / 5)

        # Prose: scenarios 1 and 5 correct = 2/5
        # (s2: prose says -3, expected 3 -> wrong)
        # (s3: prose says infinity, expected diverges -> wrong)
        # (s4: prose says 100, expected 120 -> wrong)
        assert result.prose_accuracy == pytest.approx(2 / 5)

    def test_metrics_computation(self, integration_runner):
        """Verify metrics are computed correctly from integration results."""
        result = integration_runner.run(PROBLEMS, pipelines=["symcode"])

        # Accuracy
        acc = MetricsComputer.accuracy(result.symcode_results)
        assert acc == pytest.approx(3 / 5)

        # Retry effectiveness
        eff = MetricsComputer.retry_effectiveness(result.symcode_results)
        assert eff["first_attempt_accuracy"] > 0  # at least scenario 1
        assert eff["final_accuracy"] == pytest.approx(3 / 5)
        assert eff["recovery_rate"] > 0  # scenarios 2, 4 recovered

        # Error distribution
        errors = MetricsComputer.error_distribution(result.symcode_results)
        assert "SyntaxError" in errors or "NameError" in errors

        # Token efficiency
        tok = MetricsComputer.token_efficiency(result.symcode_results)
        assert tok["avg_attempts"] > 1  # some problems needed retries
        assert tok["total_attempts"] > 5  # at least one retry

    def test_comparison_analysis(self, integration_runner):
        """Verify comparison analysis works end-to-end."""
        result = integration_runner.run(
            PROBLEMS, pipelines=["symcode", "prose"]
        )

        analyzer = ComparisonAnalyzer(
            result.symcode_results, result.prose_results
        )

        summary = analyzer.summary()
        assert summary["n"] == 5
        assert summary["symcode_accuracy"] > summary["prose_accuracy"]
        assert summary["delta_pp"] > 0

        # SymCode wins: scenarios 2, 4 (correct where prose wrong)
        assert summary["symcode_only_correct"] >= 2

        # Prose wins: scenario 5 (correct where SymCode wrong)
        assert summary["prose_only_correct"] >= 1

        stat = analyzer.statistical_test()
        assert stat["p_value"] is not None

    def test_error_taxonomy(self, integration_runner):
        """Verify error taxonomy analysis works."""
        result = integration_runner.run(PROBLEMS, pipelines=["symcode"])

        taxonomist = ErrorTaxonomist()
        report = taxonomist.analyze(result.symcode_results)

        assert report.total_problems == 5
        assert report.total_failures == 2  # scenarios 3 and 5

        # Should find code generation or runtime failures
        total_errors = sum(report.category_totals.values())
        assert total_errors > 0

    def test_retry_analysis(self, integration_runner):
        """Verify retry analysis works."""
        result = integration_runner.run(PROBLEMS, pipelines=["symcode"])

        # Correction trajectory
        trajectory = RetryAnalyzer.correction_trajectory(
            result.symcode_results, max_k=3
        )
        assert len(trajectory["cumulative_accuracy"]) == 3
        # Should show improvement from attempt 1 to attempt 2
        # (scenarios 2 and 4 are recovered on retry)

        # Correction success by error type
        by_type = RetryAnalyzer.correction_success_by_error_type(
            result.symcode_results
        )
        # Should have entries for the error types encountered
        assert len(by_type) > 0

        # Optimal k
        opt = RetryAnalyzer.optimal_k(result.symcode_results, max_k=3)
        assert opt["optimal_k"] >= 1
        assert opt["accuracy_at_optimal"] > 0

    def test_report_generation(self, integration_runner, tmp_path):
        """Verify report generation produces valid markdown."""
        result = integration_runner.run(
            PROBLEMS, pipelines=["symcode", "prose"]
        )

        report_path = tmp_path / "test_report.md"
        report = generate_report(
            result, output_path=str(report_path), title="Integration Test Report"
        )

        # Report should contain expected sections
        assert "# Integration Test Report" in report
        assert "Executive Summary" in report
        assert "Head-to-Head Comparison" in report
        assert "Per-Subject Breakdown" in report
        assert "Retry" in report
        assert "Recommendations" in report

        # File should be written
        assert report_path.exists()
        content = report_path.read_text()
        assert len(content) > 100

    def test_results_serialization(self, integration_runner, tmp_path):
        """Verify results can be saved and the JSON is valid."""
        result = integration_runner.run(
            PROBLEMS, pipelines=["symcode", "prose"]
        )

        output_path = tmp_path / "results.json"
        integration_runner.save_results(result, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())

        assert "symcode_accuracy" in data
        assert "prose_accuracy" in data
        assert "symcode_results" in data
        assert "prose_results" in data
        assert len(data["symcode_results"]) == 5
        assert len(data["prose_results"]) == 5

        # Check structure of individual results
        first = data["symcode_results"][0]
        assert "problem" in first
        assert "correct" in first
        assert "attempts" in first

    def test_concurrent_execution(self, scenario_generator, mock_executor):
        """Verify concurrent execution produces same results."""
        checker = AnswerChecker()
        retry_loop = RetryLoop(
            generator=ScenarioGenerator(),  # fresh to avoid state
            executor=mock_executor,
            checker=checker,
            max_retries=3,
        )
        runner = BenchmarkRunner(
            config={},
            generator=scenario_generator,
            retry_loop=retry_loop,
            prose_baseline=ScenarioProseBaseline(),
            checker=checker,
        )

        # Run with concurrency > 1
        result = runner.run(PROBLEMS, pipelines=["symcode"], concurrency=2)
        assert len(result.symcode_results) == 5

        # Check that results are in order (even with concurrent execution)
        for i, sr in enumerate(result.symcode_results):
            assert sr.problem == PROBLEMS[i].problem
