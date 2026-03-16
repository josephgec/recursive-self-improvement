"""Tests for interpretability evaluator: verifiability, faithfulness, readability."""

from __future__ import annotations

import pytest

from src.evaluation.interpretability import InterpretabilityEvaluator, InterpretabilityResult
from src.hybrid.pipeline import HybridResult, ReasoningStep
from src.integrative.constrained_decoder import ConstrainedOutput, Constraint


class TestInterpretabilityEvaluator:
    def test_evaluate_hybrid_results(self):
        evaluator = InterpretabilityEvaluator()
        results = [
            HybridResult(
                answer="7",
                reasoning_chain=[
                    ReasoningStep(step_type="reasoning", content="I need to compute 3 + 4."),
                    ReasoningStep(step_type="tool_call", content="Calling sympy_solve",
                                  tool_name="sympy_solve", tool_input="3 + 4"),
                    ReasoningStep(step_type="tool_result", content="The solver returned 7",
                                  tool_output="7"),
                    ReasoningStep(step_type="conclusion", content="Therefore the answer is 7."),
                ],
                tool_calls_made=1,
            ),
        ]
        result = evaluator.evaluate("hybrid", results)
        assert isinstance(result, InterpretabilityResult)
        assert result.system == "hybrid"
        assert result.step_verifiability > 0

    def test_verifiability_higher_with_tool_calls(self):
        evaluator = InterpretabilityEvaluator()
        # Result with tool calls
        with_tools = HybridResult(
            answer="7",
            reasoning_chain=[
                ReasoningStep(step_type="reasoning", content="Think"),
                ReasoningStep(step_type="tool_call", content="Call", tool_name="sympy_solve"),
                ReasoningStep(step_type="tool_result", content="7"),
                ReasoningStep(step_type="conclusion", content="7"),
            ],
        )
        # Result without tool calls
        without_tools = HybridResult(
            answer="7",
            reasoning_chain=[
                ReasoningStep(step_type="reasoning", content="Think"),
                ReasoningStep(step_type="conclusion", content="7"),
            ],
        )
        r_with = evaluator.evaluate("hybrid", [with_tools])
        r_without = evaluator.evaluate("hybrid", [without_tools])
        assert r_with.step_verifiability > r_without.step_verifiability

    def test_faithfulness_scoring(self):
        evaluator = InterpretabilityEvaluator()
        result = HybridResult(
            answer="7",
            reasoning_chain=[
                ReasoningStep(step_type="reasoning", content="Computing 3 plus 4"),
                ReasoningStep(step_type="tool_result", content="Result is 7"),
                ReasoningStep(step_type="conclusion", content="The answer is 7"),
            ],
        )
        score = evaluator._score_faithfulness(result)
        assert 0.0 <= score <= 1.0

    def test_readability_short_sentences(self):
        evaluator = InterpretabilityEvaluator()
        result = HybridResult(
            answer="7",
            reasoning_chain=[
                ReasoningStep(step_type="reasoning",
                              content="I compute 3 + 4. Therefore the answer is 7."),
                ReasoningStep(step_type="conclusion", content="7"),
            ],
        )
        score = evaluator._score_readability(result)
        assert score > 0.0

    def test_readability_with_connectives(self):
        evaluator = InterpretabilityEvaluator()
        result = HybridResult(
            answer="7",
            reasoning_chain=[
                ReasoningStep(step_type="reasoning",
                              content="Since 3 + 4 = 7, therefore the answer is 7. "
                                      "Because the sum is correct, hence we conclude."),
                ReasoningStep(step_type="conclusion", content="7"),
            ],
        )
        score = evaluator._score_readability(result)
        assert score > 0.5

    def test_integrative_verifiability(self):
        evaluator = InterpretabilityEvaluator()
        from src.integrative.pipeline import IntegrativeResult
        result = IntegrativeResult(
            answer="7",
            constrained_output=ConstrainedOutput(
                text="3 + 4 = 7",
                constraints_applied=[
                    Constraint(constraint_type="arithmetic", pattern=""),
                    Constraint(constraint_type="logical", pattern=""),
                ],
            ),
        )
        score = evaluator._score_step_verifiability(result)
        assert score > 0.3  # Constraints give partial verifiability

    def test_integrative_faithfulness(self):
        evaluator = InterpretabilityEvaluator()
        from src.integrative.pipeline import IntegrativeResult
        result = IntegrativeResult(
            answer="7",
            constrained_output=ConstrainedOutput(
                text="3 + 4 = 7",
                constraint_violations=0,
            ),
        )
        score = evaluator._score_faithfulness(result)
        assert score > 0.5

    def test_prose_minimal_verifiability(self):
        evaluator = InterpretabilityEvaluator()
        from tests.conftest import ProseResult
        result = ProseResult(answer="7")
        score = evaluator._score_step_verifiability(result)
        assert score <= 0.2

    def test_prose_faithfulness(self):
        evaluator = InterpretabilityEvaluator()
        from tests.conftest import ProseResult
        result = ProseResult(answer="7")
        score = evaluator._score_faithfulness(result)
        assert score > 0.0

    def test_readability_empty(self):
        evaluator = InterpretabilityEvaluator()
        from tests.conftest import ProseResult
        result = ProseResult(answer="")
        score = evaluator._score_readability(result)
        assert score == 0.0

    def test_overall_score_weighted(self):
        evaluator = InterpretabilityEvaluator(
            verifiability_weight=0.4,
            faithfulness_weight=0.35,
            readability_weight=0.25,
        )
        results = [
            HybridResult(
                answer="7",
                reasoning_chain=[
                    ReasoningStep(step_type="tool_call", content="Call",
                                  tool_name="sympy_solve"),
                    ReasoningStep(step_type="tool_result", content="7"),
                    ReasoningStep(step_type="conclusion",
                                  content="Therefore the answer is 7."),
                ],
            ),
        ]
        result = evaluator.evaluate("hybrid", results)
        expected = (
            0.4 * result.step_verifiability
            + 0.35 * result.faithfulness
            + 0.25 * result.readability
        )
        assert abs(result.overall_score - expected) < 1e-6

    def test_per_result_scores(self):
        evaluator = InterpretabilityEvaluator()
        results = [
            HybridResult(answer="7", reasoning_chain=[
                ReasoningStep(step_type="reasoning", content="Think."),
                ReasoningStep(step_type="conclusion", content="7"),
            ]),
            HybridResult(answer="10", reasoning_chain=[
                ReasoningStep(step_type="reasoning", content="Think."),
                ReasoningStep(step_type="conclusion", content="10"),
            ]),
        ]
        result = evaluator.evaluate("test", results)
        assert len(result.per_result_scores) == 2
        for s in result.per_result_scores:
            assert "verifiability" in s
            assert "faithfulness" in s
            assert "readability" in s

    def test_readability_long_sentences(self):
        evaluator = InterpretabilityEvaluator()
        # Create a result with very long sentences
        long = " ".join(["word"] * 50)
        result = HybridResult(
            answer="7",
            reasoning_chain=[
                ReasoningStep(step_type="reasoning", content=long),
                ReasoningStep(step_type="conclusion", content="7"),
            ],
        )
        score = evaluator._score_readability(result)
        # Long sentences should reduce readability
        assert score < 1.0
