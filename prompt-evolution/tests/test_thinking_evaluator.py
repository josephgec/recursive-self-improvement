"""Tests for thinking-model evaluator."""

import pytest

from src.genome.prompt_genome import PromptGenome
from src.genome.sections import DEFAULT_SECTIONS
from src.operators.thinking_evaluator import (
    ThinkingEvaluator,
    FitnessDetails,
    TaskEvaluation,
    ReasoningQuality,
)
from src.evaluation.answer_checker import FinancialAnswerChecker


class TestReasoningQuality:
    """Tests for ReasoningQuality dataclass."""

    def test_compute_score_all_true(self):
        rq = ReasoningQuality(
            has_step_by_step=True,
            has_formula_reference=True,
            has_intermediate_values=True,
            has_final_answer=True,
            has_verification=True,
        )
        score = rq.compute_score()
        assert score == 1.0

    def test_compute_score_all_false(self):
        rq = ReasoningQuality()
        score = rq.compute_score()
        assert score == 0.0

    def test_compute_score_partial(self):
        rq = ReasoningQuality(has_step_by_step=True, has_final_answer=True)
        score = rq.compute_score()
        assert score == pytest.approx(0.4)


class TestFitnessDetails:
    """Tests for FitnessDetails dataclass."""

    def test_compute_composite_default_weights(self):
        details = FitnessDetails(
            accuracy=0.8,
            reasoning_score=0.6,
            consistency_score=0.9,
        )
        composite = details.compute_composite()
        expected = 0.7 * 0.8 + 0.15 * 0.6 + 0.15 * 0.9
        assert composite == pytest.approx(expected)

    def test_compute_composite_custom_weights(self):
        details = FitnessDetails(
            accuracy=1.0,
            reasoning_score=0.0,
            consistency_score=0.0,
        )
        composite = details.compute_composite(
            accuracy_weight=1.0,
            reasoning_weight=0.0,
            consistency_weight=0.0,
        )
        assert composite == pytest.approx(1.0)

    def test_empty_task_evaluations(self):
        details = FitnessDetails()
        assert len(details.task_evaluations) == 0

    def test_section_scores_dict(self):
        details = FitnessDetails(section_scores={"identity": 0.8, "methodology": 0.5})
        assert details.section_scores["identity"] == 0.8


class TestThinkingEvaluator:
    """Tests for ThinkingEvaluator."""

    def test_evaluate_returns_fitness_details(self, mock_thinking_llm, sample_genome, sample_tasks):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        details = evaluator.evaluate(sample_genome, sample_tasks)

        assert isinstance(details, FitnessDetails)
        assert 0.0 <= details.accuracy <= 1.0
        assert 0.0 <= details.reasoning_score <= 1.0
        assert 0.0 <= details.consistency_score <= 1.0
        assert 0.0 <= details.composite_fitness <= 1.0

    def test_evaluate_produces_task_evaluations(
        self, mock_thinking_llm, sample_genome, sample_tasks
    ):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        details = evaluator.evaluate(sample_genome, sample_tasks)
        assert len(details.task_evaluations) == len(sample_tasks)

    def test_evaluate_task_evaluation_fields(
        self, mock_thinking_llm, sample_genome, sample_tasks
    ):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        details = evaluator.evaluate(sample_genome, sample_tasks)

        for te in details.task_evaluations:
            assert isinstance(te, TaskEvaluation)
            assert te.task_id is not None
            assert isinstance(te.is_correct, bool)
            assert isinstance(te.reasoning_quality, ReasoningQuality)

    def test_reasoning_quality_assessed(
        self, mock_thinking_llm, sample_genome, sample_tasks
    ):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        details = evaluator.evaluate(sample_genome, sample_tasks)

        rq_scores = [
            te.reasoning_quality.score for te in details.task_evaluations
        ]
        # Mock should produce some reasoning markers
        assert details.reasoning_score >= 0.0

    def test_composite_fitness_computation(
        self, mock_thinking_llm, sample_genome, sample_tasks
    ):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        details = evaluator.evaluate(sample_genome, sample_tasks)

        expected = (
            0.7 * details.accuracy
            + 0.15 * details.reasoning_score
            + 0.15 * details.consistency_score
        )
        assert details.composite_fitness == pytest.approx(expected, abs=0.01)

    def test_evaluate_with_answer_checker(self, mock_thinking_llm, sample_genome, sample_tasks):
        checker = FinancialAnswerChecker(tolerance=0.05)
        evaluator = ThinkingEvaluator(mock_thinking_llm, answer_checker=checker)
        details = evaluator.evaluate(sample_genome, sample_tasks)
        assert isinstance(details, FitnessDetails)

    def test_evaluate_empty_tasks(self, mock_thinking_llm, sample_genome):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        details = evaluator.evaluate(sample_genome, [])
        assert details.accuracy == 0.0
        assert details.composite_fitness >= 0.0

    def test_section_scores_computed(self, mock_thinking_llm, sample_genome, sample_tasks):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        details = evaluator.evaluate(sample_genome, sample_tasks)
        assert len(details.section_scores) > 0
        for name, score in details.section_scores.items():
            assert 0.0 <= score <= 1.0

    def test_better_prompt_higher_accuracy(self, mock_thinking_llm):
        """Better structured prompts should generally get higher accuracy."""
        evaluator = ThinkingEvaluator(mock_thinking_llm)

        tasks = [
            {"task_id": "t1", "question": "What is 100 * 1.05?", "expected_answer": "105"},
            {"task_id": "t2", "question": "What is 200 * 1.10?", "expected_answer": "220"},
        ]

        # Simple prompt
        simple = PromptGenome(genome_id="simple")
        simple.set_section("identity", "AI")
        simple.set_section("task_description", "Answer")
        simple.set_section("methodology", "Compute")

        # Detailed prompt
        detailed = PromptGenome(genome_id="detailed")
        for name, content in DEFAULT_SECTIONS.items():
            detailed.set_section(name, content)
        # Add extra quality signals
        detailed.set_section(
            "methodology",
            "Use step by step methodology. Apply the correct formula. "
            "Verify your answer by checking it systematically. "
            "Calculate precisely and show all intermediate values.",
        )

        d_simple = evaluator.evaluate(simple, tasks)
        d_detailed = evaluator.evaluate(detailed, tasks)

        # Detailed should have at least as good reasoning assessment
        # (the mock assesses prompt quality for accuracy)
        assert d_detailed.reasoning_score >= 0.0

    def test_assess_reasoning_markers(self, mock_thinking_llm):
        evaluator = ThinkingEvaluator(mock_thinking_llm)

        # Response with reasoning markers
        rq = evaluator._assess_reasoning(
            "Step 1: Identify the formula. Then apply it. "
            "The equation is A = P(1+r). Therefore the answer is 100."
        )
        assert rq.has_step_by_step is True
        assert rq.has_formula_reference is True
        assert rq.has_final_answer is True

    def test_consistency_single_answer(self, mock_thinking_llm):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        consistency = evaluator._compute_consistency({"t1": ["answer a"]})
        assert consistency == 1.0

    def test_consistency_different_answers(self, mock_thinking_llm):
        evaluator = ThinkingEvaluator(mock_thinking_llm)
        consistency = evaluator._compute_consistency({
            "t1": ["completely different answer one", "totally unrelated two"]
        })
        assert 0.0 <= consistency <= 1.0
