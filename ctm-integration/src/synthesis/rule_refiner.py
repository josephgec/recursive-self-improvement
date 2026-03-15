"""Rule refinement based on failure analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from src.synthesis.candidate_generator import CandidateGenerator, CandidateRule, IOExample
from src.synthesis.empirical_verifier import EmpiricalVerifier, FailedExample, VerificationResult


class RuleRefiner:
    """Refines rules based on failure case analysis.

    Strategies:
    1. Add special cases for failures
    2. Generate completely new rules
    3. Combine successful partial rules
    """

    def __init__(
        self,
        generator: Optional[CandidateGenerator] = None,
        verifier: Optional[EmpiricalVerifier] = None,
    ) -> None:
        self.generator = generator or CandidateGenerator()
        self.verifier = verifier or EmpiricalVerifier()

    def refine(
        self,
        rule: CandidateRule,
        failures: List[FailedExample],
        examples: List[IOExample],
    ) -> List[CandidateRule]:
        """Refine a rule based on its failures.

        Args:
            rule: The rule to refine.
            failures: List of failed examples.
            examples: All original examples.

        Returns:
            List of refined candidate rules.
        """
        if not failures:
            return [rule]

        # Analyze failure patterns
        analysis = self._analyze_failures(failures)

        # Generate refinements
        failure_tuples = [
            (f.input, f.expected_output, f.actual_output) for f in failures
        ]
        refined = self.generator.generate_refinement(rule, failure_tuples, examples)

        # Verify refinements don't regress
        valid_refinements = []
        for candidate in refined:
            if self._ensure_no_regression(candidate, rule, examples):
                valid_refinements.append(candidate)

        # If no valid refinements, return the originals anyway
        if not valid_refinements:
            valid_refinements = refined

        return valid_refinements

    def _analyze_failures(
        self, failures: List[FailedExample]
    ) -> dict:
        """Analyze patterns in failure cases.

        Args:
            failures: List of failed examples.

        Returns:
            Analysis dictionary with failure patterns.
        """
        analysis = {
            "total_failures": len(failures),
            "error_failures": 0,
            "wrong_output_failures": 0,
            "patterns": [],
        }

        for f in failures:
            if f.error:
                analysis["error_failures"] += 1
            else:
                analysis["wrong_output_failures"] += 1

        # Check for common patterns
        if all(f.actual_output is None for f in failures):
            analysis["patterns"].append("all_none_outputs")
        elif all(f.error for f in failures):
            analysis["patterns"].append("all_errors")

        return analysis

    def _ensure_no_regression(
        self,
        refined: CandidateRule,
        original: CandidateRule,
        examples: List[IOExample],
    ) -> bool:
        """Check that the refined rule doesn't regress on examples the original got right.

        Args:
            refined: The refined rule.
            original: The original rule.
            examples: All examples.

        Returns:
            True if the refined rule is at least as good as the original.
        """
        original_result = self.verifier.verify(original, examples)
        refined_result = self.verifier.verify(refined, examples)

        return refined_result.accuracy >= original_result.accuracy
