"""Tests for rule refinement."""

import pytest

from src.synthesis.candidate_generator import CandidateRule, IOExample
from src.synthesis.empirical_verifier import FailedExample
from src.synthesis.rule_refiner import RuleRefiner


class TestRuleRefiner:
    """Tests for the rule refiner."""

    def test_refine_with_failures(self):
        """Refining with failures should produce new candidates."""
        refiner = RuleRefiner()
        rule = CandidateRule(
            rule_id="base",
            source_code="def rule(x):\n    return x + 1\n",
            description="Wrong rule",
            domain="math",
        )
        failures = [
            FailedExample(input=2, expected_output=4, actual_output=3),
            FailedExample(input=5, expected_output=10, actual_output=6),
        ]
        examples = [
            IOExample(input=1, output=2, domain="math"),
            IOExample(input=2, output=4, domain="math"),
            IOExample(input=5, output=10, domain="math"),
        ]
        refined = refiner.refine(rule, failures, examples)
        assert len(refined) > 0

    def test_refine_no_failures(self):
        """Refining with no failures should return original."""
        refiner = RuleRefiner()
        rule = CandidateRule(
            rule_id="perfect",
            source_code="def rule(x):\n    return x * 2\n",
            description="Perfect",
            domain="math",
        )
        refined = refiner.refine(rule, [], [])
        assert len(refined) == 1
        assert refined[0].rule_id == "perfect"

    def test_analyze_failures(self):
        """Failure analysis should categorize failures."""
        refiner = RuleRefiner()
        failures = [
            FailedExample(input=1, expected_output=2, actual_output=3),
            FailedExample(input=2, expected_output=4, actual_output=None, error="crash"),
        ]
        analysis = refiner._analyze_failures(failures)
        assert analysis["total_failures"] == 2
        assert analysis["error_failures"] == 1
        assert analysis["wrong_output_failures"] == 1

    def test_analyze_all_errors(self):
        """All-error pattern should be detected."""
        refiner = RuleRefiner()
        failures = [
            FailedExample(input=1, expected_output=2, actual_output="err", error="e1"),
            FailedExample(input=2, expected_output=4, actual_output="err", error="e2"),
        ]
        analysis = refiner._analyze_failures(failures)
        assert "all_errors" in analysis["patterns"]

    def test_analyze_all_none_outputs(self):
        """All-None pattern should be detected."""
        refiner = RuleRefiner()
        failures = [
            FailedExample(input=1, expected_output=2, actual_output=None),
            FailedExample(input=2, expected_output=4, actual_output=None),
        ]
        analysis = refiner._analyze_failures(failures)
        assert "all_none_outputs" in analysis["patterns"]

    def test_ensure_no_regression_pass(self):
        """No-regression check should pass for better rule."""
        refiner = RuleRefiner()
        examples = [
            IOExample(input=1, output=2),
            IOExample(input=2, output=4),
        ]
        original = CandidateRule(
            rule_id="orig",
            source_code="def rule(x):\n    return x + 1\n",
            description="wrong",
        )
        refined = CandidateRule(
            rule_id="ref",
            source_code="def rule(x):\n    return x * 2\n",
            description="correct",
        )
        assert refiner._ensure_no_regression(refined, original, examples)
