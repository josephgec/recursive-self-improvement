"""Tests for benchmark loading, metrics, and comparison analysis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.benchmarks.math500 import MATH500Loader, BenchmarkProblem, _extract_boxed
from src.benchmarks.olympiad import OlympiadBenchLoader
from src.benchmarks.metrics import MetricsComputer
from src.benchmarks.comparison import ComparisonAnalyzer
from src.verification.result_types import (
    AttemptRecord,
    CodeError,
    CodeExecutionResult,
    SolveResult,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_solve_result(
    correct: bool,
    final_answer: str | None = None,
    expected_answer: str = "42",
    num_attempts: int = 1,
    task_type: str = "algebra",
    pipeline: str = "symcode",
    total_time: float = 1.0,
    attempts: list[AttemptRecord] | None = None,
) -> SolveResult:
    """Build a SolveResult for testing."""
    if attempts is None:
        attempts = []
        for i in range(num_attempts):
            is_last = i == num_attempts - 1
            a_correct = correct if is_last else False
            attempts.append(
                AttemptRecord(
                    attempt_number=i + 1,
                    code="answer = 42",
                    execution_result=CodeExecutionResult(
                        success=True,
                        stdout="Answer: 42",
                        answer="42",
                    ),
                    extracted_answer=final_answer if is_last else "wrong",
                    answer_correct=a_correct,
                )
            )
    return SolveResult(
        problem="What is 6 * 7?",
        expected_answer=expected_answer,
        final_answer=final_answer,
        correct=correct,
        num_attempts=num_attempts,
        attempts=attempts,
        task_type=task_type,
        pipeline=pipeline,
        total_time=total_time,
    )


# ── MATH500Loader tests ────────────────────────────────────────────


class TestExtractBoxed:
    def test_simple(self):
        assert _extract_boxed(r"The answer is \boxed{42}.") == "42"

    def test_nested_braces(self):
        assert _extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_no_boxed(self):
        assert _extract_boxed("No boxed content here") is None

    def test_multiple_boxed(self):
        text = r"First \boxed{1} then \boxed{2}"
        # Should return the last one
        assert _extract_boxed(text) == "2"

    def test_empty_boxed(self):
        assert _extract_boxed(r"\boxed{}") == ""


class TestMATH500Loader:
    def test_load_success(self):
        """Test loading with a mock dataset."""
        mock_ds = [
            {
                "problem": "What is 2+3?",
                "solution": r"We compute $2+3 = \boxed{5}$.",
                "type": "Prealgebra",
                "level": "Level 1",
            },
            {
                "problem": "Solve x^2 = 4",
                "solution": r"$x = \pm 2$. The answer is $\boxed{2}$.",
                "type": "Algebra",
                "level": "Level 2",
            },
            {
                "problem": "Find the GCD of 12 and 8",
                "solution": r"$\gcd(12,8) = \boxed{4}$.",
                "type": "Number Theory",
                "level": "Level 3",
            },
        ]

        mock_load = MagicMock(return_value=mock_ds)
        loader = MATH500Loader(num_problems=10)

        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys
            sys.modules["datasets"].load_dataset = mock_load
            problems = loader.load()

        assert len(problems) == 3
        assert problems[0].expected_answer == "5"
        assert problems[0].subject == "prealgebra"
        assert problems[0].difficulty == 1
        assert problems[1].expected_answer == "2"
        assert problems[1].difficulty == 2

    def test_load_by_subject(self):
        """Test filtering by subject."""
        mock_ds = [
            {
                "problem": "p1",
                "solution": r"\boxed{1}",
                "type": "Algebra",
                "level": "Level 1",
            },
            {
                "problem": "p2",
                "solution": r"\boxed{2}",
                "type": "Geometry",
                "level": "Level 2",
            },
        ]

        mock_load = MagicMock(return_value=mock_ds)
        loader = MATH500Loader()
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys
            sys.modules["datasets"].load_dataset = mock_load
            algebra = loader.load_by_subject("algebra")

        assert len(algebra) == 1
        assert algebra[0].subject == "algebra"

    def test_load_by_difficulty(self):
        """Test filtering by difficulty."""
        mock_ds = [
            {
                "problem": "p1",
                "solution": r"\boxed{1}",
                "type": "Algebra",
                "level": "Level 1",
            },
            {
                "problem": "p2",
                "solution": r"\boxed{2}",
                "type": "Algebra",
                "level": "Level 3",
            },
        ]

        mock_load = MagicMock(return_value=mock_ds)
        loader = MATH500Loader()
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys
            sys.modules["datasets"].load_dataset = mock_load
            level3 = loader.load_by_difficulty(3)

        assert len(level3) == 1
        assert level3[0].difficulty == 3

    def test_load_graceful_failure(self):
        """Test that load returns empty list when dataset unavailable."""
        loader = MATH500Loader(dataset_name="nonexistent/dataset")
        # Simulate import succeeding but load_dataset raising
        mock_mod = MagicMock()
        mock_mod.load_dataset = MagicMock(side_effect=Exception("Dataset not found"))
        with patch.dict("sys.modules", {"datasets": mock_mod}):
            problems = loader.load()
        assert problems == []


class TestOlympiadBenchLoader:
    def test_synthetic_fallback(self):
        """Test that synthetic problems are loaded when HF is unavailable."""
        loader = OlympiadBenchLoader()
        mock_mod = MagicMock()
        mock_mod.load_dataset = MagicMock(side_effect=Exception("Not found"))
        with patch.dict("sys.modules", {"datasets": mock_mod}):
            problems = loader.load()

        assert len(problems) > 0
        assert all(isinstance(p, BenchmarkProblem) for p in problems)
        assert problems[0].problem_id.startswith("olympiad_synth_")

    def test_num_problems_limit(self):
        """Test limiting number of synthetic problems."""
        loader = OlympiadBenchLoader(num_problems=3)
        mock_mod = MagicMock()
        mock_mod.load_dataset = MagicMock(side_effect=Exception("Not found"))
        with patch.dict("sys.modules", {"datasets": mock_mod}):
            problems = loader.load()

        assert len(problems) == 3


# ── MetricsComputer tests ──────────────────────────────────────────


class TestMetricsComputer:
    def test_accuracy(self):
        results = [
            _make_solve_result(correct=True),
            _make_solve_result(correct=True),
            _make_solve_result(correct=False),
            _make_solve_result(correct=False),
            _make_solve_result(correct=True),
        ]
        assert MetricsComputer.accuracy(results) == pytest.approx(0.6)

    def test_accuracy_empty(self):
        assert MetricsComputer.accuracy([]) == 0.0

    def test_pass_at_k_basic(self):
        # If n=3, c=1, k=1: 1 - C(2,1)/C(3,1) = 1 - 2/3 = 1/3
        assert MetricsComputer.pass_at_k(3, 1, 1) == pytest.approx(1 / 3)

    def test_pass_at_k_all_correct(self):
        assert MetricsComputer.pass_at_k(5, 5, 1) == pytest.approx(1.0)

    def test_pass_at_k_none_correct(self):
        assert MetricsComputer.pass_at_k(5, 0, 1) == pytest.approx(0.0)

    def test_pass_at_k_k_equals_n(self):
        # n=3, c=1, k=3: should be 1.0 (at least one correct in all 3)
        assert MetricsComputer.pass_at_k(3, 1, 3) == pytest.approx(1.0)

    def test_accuracy_by_subject(self):
        results = [
            _make_solve_result(correct=True, task_type="algebra"),
            _make_solve_result(correct=False, task_type="algebra"),
            _make_solve_result(correct=True, task_type="geometry"),
        ]
        breakdown = MetricsComputer.accuracy_by_subject(results)
        assert breakdown["algebra"]["accuracy"] == pytest.approx(0.5)
        assert breakdown["geometry"]["accuracy"] == pytest.approx(1.0)
        assert breakdown["algebra"]["total"] == 2
        assert breakdown["geometry"]["total"] == 1

    def test_accuracy_by_difficulty(self):
        results = [
            _make_solve_result(correct=True),
            _make_solve_result(correct=False),
            _make_solve_result(correct=True),
        ]
        diffs = [1, 1, 3]
        breakdown = MetricsComputer.accuracy_by_difficulty(results, diffs)
        assert breakdown[1]["accuracy"] == pytest.approx(0.5)
        assert breakdown[3]["accuracy"] == pytest.approx(1.0)

    def test_retry_effectiveness(self):
        # One first-attempt correct, one self-corrected, one always wrong
        attempt_correct_first = AttemptRecord(
            attempt_number=1,
            code="x = 42",
            execution_result=CodeExecutionResult(success=True, answer="42"),
            extracted_answer="42",
            answer_correct=True,
        )
        attempt_wrong = AttemptRecord(
            attempt_number=1,
            code="x = 0",
            execution_result=CodeExecutionResult(success=True, answer="0"),
            extracted_answer="0",
            answer_correct=False,
        )
        attempt_corrected = AttemptRecord(
            attempt_number=2,
            code="x = 42",
            execution_result=CodeExecutionResult(success=True, answer="42"),
            extracted_answer="42",
            answer_correct=True,
        )
        attempt_still_wrong = AttemptRecord(
            attempt_number=2,
            code="x = 1",
            execution_result=CodeExecutionResult(success=True, answer="1"),
            extracted_answer="1",
            answer_correct=False,
        )

        results = [
            _make_solve_result(
                correct=True,
                final_answer="42",
                num_attempts=1,
                attempts=[attempt_correct_first],
            ),
            _make_solve_result(
                correct=True,
                final_answer="42",
                num_attempts=2,
                attempts=[attempt_wrong, attempt_corrected],
            ),
            _make_solve_result(
                correct=False,
                final_answer="1",
                num_attempts=2,
                attempts=[attempt_wrong, attempt_still_wrong],
            ),
        ]

        eff = MetricsComputer.retry_effectiveness(results)
        assert eff["first_attempt_accuracy"] == pytest.approx(1 / 3)
        assert eff["final_accuracy"] == pytest.approx(2 / 3)
        assert eff["recovery_rate"] == pytest.approx(0.5)  # 1 of 2 first-wrongs recovered

    def test_error_distribution(self):
        err1 = CodeError(error_type="SyntaxError", message="bad syntax")
        err2 = CodeError(error_type="NameError", message="x not defined")

        results = [
            _make_solve_result(
                correct=False,
                attempts=[
                    AttemptRecord(
                        attempt_number=1,
                        code="",
                        execution_result=CodeExecutionResult(
                            success=False, error=err1
                        ),
                    ),
                    AttemptRecord(
                        attempt_number=2,
                        code="",
                        execution_result=CodeExecutionResult(
                            success=False, error=err2
                        ),
                    ),
                ],
            ),
            _make_solve_result(
                correct=False,
                attempts=[
                    AttemptRecord(
                        attempt_number=1,
                        code="",
                        execution_result=CodeExecutionResult(
                            success=False, error=err1
                        ),
                    ),
                ],
            ),
        ]

        dist = MetricsComputer.error_distribution(results)
        assert dist["SyntaxError"] == 2
        assert dist["NameError"] == 1

    def test_token_efficiency(self):
        results = [
            _make_solve_result(correct=True, num_attempts=1),
            _make_solve_result(correct=True, num_attempts=3),
            _make_solve_result(correct=False, num_attempts=3),
        ]
        eff = MetricsComputer.token_efficiency(results)
        assert eff["avg_attempts"] == pytest.approx(7 / 3)
        assert eff["total_attempts"] == 7
        assert eff["attempts_per_correct"] == pytest.approx(7 / 2)

    def test_latency_stats(self):
        results = [
            _make_solve_result(correct=True, total_time=1.0),
            _make_solve_result(correct=True, total_time=2.0),
            _make_solve_result(correct=True, total_time=3.0),
        ]
        stats = MetricsComputer.latency_stats(results)
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["median"] == pytest.approx(2.0)
        assert stats["total"] == pytest.approx(6.0)


# ── ComparisonAnalyzer tests ───────────────────────────────────────


class TestComparisonAnalyzer:
    def _make_paired_results(self):
        """Create synthetic paired results for comparison testing."""
        symcode = [
            _make_solve_result(correct=True, task_type="algebra"),  # both correct
            _make_solve_result(correct=True, task_type="algebra"),  # symcode only
            _make_solve_result(correct=False, task_type="geometry"),  # prose only
            _make_solve_result(correct=False, task_type="geometry"),  # both wrong
            _make_solve_result(correct=True, task_type="number_theory"),  # symcode only
        ]
        prose = [
            _make_solve_result(correct=True, task_type="algebra", pipeline="prose"),
            _make_solve_result(correct=False, task_type="algebra", pipeline="prose"),
            _make_solve_result(correct=True, task_type="geometry", pipeline="prose"),
            _make_solve_result(correct=False, task_type="geometry", pipeline="prose"),
            _make_solve_result(correct=False, task_type="number_theory", pipeline="prose"),
        ]
        return symcode, prose

    def test_summary(self):
        symcode, prose = self._make_paired_results()
        analyzer = ComparisonAnalyzer(symcode, prose)
        summary = analyzer.summary()

        assert summary["symcode_accuracy"] == pytest.approx(3 / 5)
        assert summary["prose_accuracy"] == pytest.approx(2 / 5)
        assert summary["delta_pp"] == pytest.approx(20.0)
        assert summary["symcode_only_correct"] == 2
        assert summary["prose_only_correct"] == 1
        assert summary["both_correct"] == 1
        assert summary["both_wrong"] == 1
        assert summary["n"] == 5
        assert summary["mcnemar_p_value"] is not None

    def test_per_subject_comparison(self):
        symcode, prose = self._make_paired_results()
        analyzer = ComparisonAnalyzer(symcode, prose)
        by_subject = analyzer.per_subject_comparison()

        assert "algebra" in by_subject
        assert by_subject["algebra"]["symcode_accuracy"] == pytest.approx(1.0)
        assert by_subject["algebra"]["prose_accuracy"] == pytest.approx(0.5)
        assert by_subject["geometry"]["symcode_accuracy"] == pytest.approx(0.0)
        assert by_subject["geometry"]["prose_accuracy"] == pytest.approx(0.5)

    def test_qualitative_examples(self):
        symcode, prose = self._make_paired_results()
        analyzer = ComparisonAnalyzer(symcode, prose)
        examples = analyzer.qualitative_examples()

        assert len(examples["symcode_wins"]) == 2
        assert len(examples["prose_wins"]) == 1
        assert len(examples["both_fail"]) == 1

    def test_statistical_test(self):
        symcode, prose = self._make_paired_results()
        analyzer = ComparisonAnalyzer(symcode, prose)
        stat = analyzer.statistical_test()

        assert stat["b_symcode_only"] == 2
        assert stat["c_prose_only"] == 1
        assert stat["p_value"] is not None
        assert isinstance(stat["significant_005"], bool)

    def test_failure_mode_comparison(self):
        symcode, prose = self._make_paired_results()
        analyzer = ComparisonAnalyzer(symcode, prose)
        failure = analyzer.failure_mode_comparison()

        assert "symcode_failure_modes" in failure
        assert "prose_failure_modes" in failure

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            ComparisonAnalyzer(
                [_make_solve_result(correct=True)],
                [_make_solve_result(correct=True), _make_solve_result(correct=False)],
            )

    def test_failure_mode_with_errors(self):
        """Verify failure modes categorize no_answer, execution_error, wrong_answer."""
        err = CodeError(error_type="SyntaxError", message="bad syntax")

        # SymCode: no answer, execution error, wrong answer
        symcode = [
            _make_solve_result(
                correct=False,
                final_answer=None,
                task_type="algebra",
                attempts=[
                    AttemptRecord(
                        attempt_number=1,
                        code="",
                        execution_result=CodeExecutionResult(success=True),
                        answer_correct=False,
                    ),
                ],
            ),
            _make_solve_result(
                correct=False,
                final_answer="wrong",
                task_type="algebra",
                attempts=[
                    AttemptRecord(
                        attempt_number=1,
                        code="",
                        execution_result=CodeExecutionResult(
                            success=False, error=err
                        ),
                    ),
                ],
            ),
            _make_solve_result(
                correct=False,
                final_answer="wrong",
                task_type="algebra",
                attempts=[
                    AttemptRecord(
                        attempt_number=1,
                        code="",
                        execution_result=CodeExecutionResult(success=True),
                        answer_correct=False,
                    ),
                ],
            ),
        ]
        prose = [
            _make_solve_result(
                correct=False,
                final_answer=None,
                task_type="algebra",
                pipeline="prose",
            ),
            _make_solve_result(
                correct=False,
                final_answer="wrong",
                task_type="algebra",
                pipeline="prose",
            ),
            _make_solve_result(
                correct=True,
                final_answer="42",
                task_type="algebra",
                pipeline="prose",
            ),
        ]

        analyzer = ComparisonAnalyzer(symcode, prose)
        failure = analyzer.failure_mode_comparison()

        assert failure["symcode_failure_modes"]["no_answer"] == 1
        assert failure["symcode_failure_modes"]["execution_error"] == 1
        assert failure["symcode_failure_modes"]["wrong_answer"] == 1
        assert failure["prose_failure_modes"]["no_answer"] == 1
        assert failure["prose_failure_modes"]["wrong_answer"] == 1

    def test_per_difficulty_comparison(self):
        """Test per-difficulty comparison breakdown."""
        symcode = [
            _make_solve_result(correct=True, task_type="algebra"),
            _make_solve_result(correct=False, task_type="algebra"),
            _make_solve_result(correct=True, task_type="geometry"),
        ]
        prose = [
            _make_solve_result(correct=False, task_type="algebra", pipeline="prose"),
            _make_solve_result(correct=True, task_type="algebra", pipeline="prose"),
            _make_solve_result(correct=True, task_type="geometry", pipeline="prose"),
        ]

        analyzer = ComparisonAnalyzer(symcode, prose)
        by_diff = analyzer.per_difficulty_comparison(difficulties=[1, 1, 2])

        assert 1 in by_diff
        assert 2 in by_diff
        assert by_diff[1]["count"] == 2
        assert by_diff[2]["count"] == 1

    def test_qualitative_max_examples(self):
        """max_examples should limit returned examples."""
        symcode = [
            _make_solve_result(correct=True, task_type="algebra"),
        ] * 10
        prose = [
            _make_solve_result(correct=False, task_type="algebra", pipeline="prose"),
        ] * 10

        analyzer = ComparisonAnalyzer(symcode, prose)
        examples = analyzer.qualitative_examples(max_examples=3)
        assert len(examples["symcode_wins"]) <= 3

    def test_mcnemar_zero_discordant(self):
        """McNemar with zero discordant pairs should return p=1.0."""
        symcode = [_make_solve_result(correct=True)]
        prose = [_make_solve_result(correct=True, pipeline="prose")]
        analyzer = ComparisonAnalyzer(symcode, prose)
        result = analyzer.statistical_test()
        assert result["p_value"] == pytest.approx(1.0)


class TestOlympiadBenchLoaderExtended:
    """Additional tests for OlympiadBench loader."""

    def test_load_by_subject(self):
        """Test filtering synthetic problems by subject."""
        loader = OlympiadBenchLoader()
        mock_mod = MagicMock()
        mock_mod.load_dataset = MagicMock(side_effect=Exception("Not found"))
        with patch.dict("sys.modules", {"datasets": mock_mod}):
            algebra = loader.load_by_subject("algebra")

        assert len(algebra) > 0
        assert all(p.subject == "algebra" for p in algebra)

    def test_load_by_difficulty(self):
        """Test filtering synthetic problems by difficulty."""
        loader = OlympiadBenchLoader()
        mock_mod = MagicMock()
        mock_mod.load_dataset = MagicMock(side_effect=Exception("Not found"))
        with patch.dict("sys.modules", {"datasets": mock_mod}):
            hard = loader.load_by_difficulty(5)

        assert len(hard) > 0
        assert all(p.difficulty == 5 for p in hard)

    def test_synthetic_problems_have_required_fields(self):
        """All synthetic problems should have problem, answer, subject."""
        loader = OlympiadBenchLoader()
        mock_mod = MagicMock()
        mock_mod.load_dataset = MagicMock(side_effect=Exception("Not found"))
        with patch.dict("sys.modules", {"datasets": mock_mod}):
            problems = loader.load()

        for p in problems:
            assert p.problem
            assert p.expected_answer
            assert p.subject
            assert p.difficulty > 0

    def test_huggingface_success(self):
        """Test loading from mocked HuggingFace dataset."""
        mock_ds = [
            {
                "problem": "Find x if x^2 = 16",
                "answer": "4",
                "subject": "algebra",
                "difficulty": 2,
                "solution": "x = 4",
            },
            {
                "question": "What is pi?",
                "expected_answer": "3.14",
                "type": "geometry",
                "level": 1,
                "solution": "",
            },
        ]
        mock_mod = MagicMock()
        mock_mod.load_dataset = MagicMock(return_value=mock_ds)

        loader = OlympiadBenchLoader(num_problems=10)
        with patch.dict("sys.modules", {"datasets": mock_mod}):
            problems = loader.load()

        assert len(problems) == 2
        assert problems[0].expected_answer == "4"
        assert problems[0].subject == "algebra"
        assert problems[1].expected_answer == "3.14"
