"""Tests for empirical verification of rules."""

import pytest

from src.synthesis.candidate_generator import CandidateRule, IOExample
from src.synthesis.empirical_verifier import EmpiricalVerifier, VerificationResult, FailedExample


class TestVerification:
    """Tests for rule verification."""

    def test_verify_correct_rule(self, verifier, double_examples):
        """Correct rule should pass all examples."""
        rule = CandidateRule(
            rule_id="correct",
            source_code="def rule(x):\n    return x * 2\n",
            description="Doubles input",
        )
        result = verifier.verify(rule, double_examples)
        assert isinstance(result, VerificationResult)
        assert result.accuracy == 1.0
        assert result.is_perfect
        assert result.passed == len(double_examples)
        assert len(result.failures) == 0

    def test_verify_incorrect_rule(self, verifier, double_examples):
        """Incorrect rule should have failures."""
        rule = CandidateRule(
            rule_id="wrong",
            source_code="def rule(x):\n    return x + 1\n",
            description="Wrong rule",
        )
        result = verifier.verify(rule, double_examples)
        assert result.accuracy < 1.0
        assert len(result.failures) > 0
        assert not result.is_perfect

    def test_verify_partial_rule(self, verifier):
        """Partially correct rule should show partial accuracy."""
        examples = [
            IOExample(input=1, output=2),
            IOExample(input=2, output=4),
            IOExample(input=0, output=1),  # x+1=1, 2*0=0 -> fails for x*2
        ]
        rule = CandidateRule(
            rule_id="partial",
            source_code="def rule(x):\n    return x * 2\n",
            description="Doubles",
        )
        result = verifier.verify(rule, examples)
        assert result.is_partial
        assert 0 < result.accuracy < 1.0

    def test_verify_crashing_rule(self, verifier, double_examples):
        """Rule that raises exceptions should fail gracefully."""
        rule = CandidateRule(
            rule_id="crash",
            source_code="def rule(x):\n    raise ValueError('boom')\n",
            description="Crashing rule",
        )
        result = verifier.verify(rule, double_examples)
        assert result.accuracy == 0.0
        assert all(f.error is not None for f in result.failures)

    def test_verify_syntax_error_rule(self, verifier, double_examples):
        """Rule with syntax error should fail to compile."""
        rule = CandidateRule(
            rule_id="syntax_err",
            source_code="def rule(x)\n    return x * 2\n",  # missing colon
            description="Syntax error",
        )
        result = verifier.verify(rule, double_examples)
        assert result.accuracy == 0.0
        assert result.error is not None

    def test_verify_list_input(self, verifier, sum_examples):
        """Rules operating on lists should work."""
        rule = CandidateRule(
            rule_id="list_sum",
            source_code="def rule(x):\n    return sum(x)\n",
            description="Sum of list",
        )
        result = verifier.verify(rule, sum_examples)
        assert result.accuracy == 1.0

    def test_failure_details(self, verifier, double_examples):
        """Failed examples should include expected and actual outputs."""
        rule = CandidateRule(
            rule_id="wrong",
            source_code="def rule(x):\n    return x + 1\n",
            description="Wrong",
        )
        result = verifier.verify(rule, double_examples)
        for f in result.failures:
            assert isinstance(f, FailedExample)
            assert f.expected_output is not None
            # actual_output is the wrong answer, not None
            assert f.actual_output is not None or f.error is not None

    def test_verify_empty_examples(self, verifier):
        """Verifying with empty examples should return 0.0 accuracy."""
        rule = CandidateRule(
            rule_id="any",
            source_code="def rule(x):\n    return x\n",
            description="Identity",
        )
        result = verifier.verify(rule, [])
        assert result.accuracy == 0.0
        assert result.total == 0


class TestGeneralization:
    """Tests for generalization verification."""

    def test_verify_generalization(self, verifier):
        """Test generalization from train to test examples."""
        rule = CandidateRule(
            rule_id="double",
            source_code="def rule(x):\n    return x * 2\n",
            description="Doubles",
        )
        train = [
            IOExample(input=1, output=2),
            IOExample(input=2, output=4),
        ]
        test = [
            IOExample(input=5, output=10),
            IOExample(input=100, output=200),
        ]
        result = verifier.verify_generalization(rule, train, test)
        assert result["train"].accuracy == 1.0
        assert result["test"].accuracy == 1.0
        assert result["generalizes"]

    def test_overfitting_detection(self, verifier):
        """Overfit rule should not generalize."""
        # Rule works only for specific inputs via lookup
        rule = CandidateRule(
            rule_id="overfit",
            source_code=(
                "def rule(x):\n"
                "    mapping = {1: 2, 2: 4}\n"
                "    return mapping.get(x, 0)\n"
            ),
            description="Lookup only",
        )
        train = [
            IOExample(input=1, output=2),
            IOExample(input=2, output=4),
        ]
        test = [
            IOExample(input=5, output=10),
            IOExample(input=100, output=200),
        ]
        result = verifier.verify_generalization(rule, train, test)
        assert result["train"].accuracy == 1.0
        assert result["test"].accuracy < 1.0
