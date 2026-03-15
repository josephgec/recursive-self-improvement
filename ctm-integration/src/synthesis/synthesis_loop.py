"""Main synthesis loop: iteratively generate, verify, rank, and select rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.bdm.scorer import BDMScorer
from src.synthesis.candidate_generator import CandidateGenerator, CandidateRule, IOExample
from src.synthesis.complexity_ranker import ComplexityRanker
from src.synthesis.empirical_verifier import EmpiricalVerifier, VerificationResult
from src.synthesis.pareto_selector import ParetoSelector, ScoredRule
from src.synthesis.rule_refiner import RuleRefiner


@dataclass
class IterationResult:
    """Result of a single synthesis iteration."""

    iteration: int
    candidates_generated: int
    candidates_verified: int
    best_accuracy: float
    best_rule: Optional[CandidateRule] = None
    pareto_front_size: int = 0
    scored_rules: List[ScoredRule] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Result of the full synthesis loop."""

    iterations: List[IterationResult] = field(default_factory=list)
    best_rules: List[ScoredRule] = field(default_factory=list)
    total_candidates: int = 0
    total_iterations: int = 0
    final_best_accuracy: float = 0.0

    @property
    def best_rule(self) -> Optional[ScoredRule]:
        if self.best_rules:
            return max(self.best_rules, key=lambda r: r.accuracy)
        return None


class SymbolicSynthesisLoop:
    """Iterative synthesis loop that generates, verifies, ranks, and selects rules.

    Each iteration:
    1. Generate candidate rules
    2. Verify candidates against examples
    3. Rank by complexity (BDM/MDL)
    4. Select Pareto-optimal rules
    5. Refine best candidates for next iteration
    """

    def __init__(
        self,
        generator: Optional[CandidateGenerator] = None,
        verifier: Optional[EmpiricalVerifier] = None,
        ranker: Optional[ComplexityRanker] = None,
        selector: Optional[ParetoSelector] = None,
        refiner: Optional[RuleRefiner] = None,
        scorer: Optional[BDMScorer] = None,
    ) -> None:
        self.scorer = scorer or BDMScorer()
        self.generator = generator or CandidateGenerator()
        self.verifier = verifier or EmpiricalVerifier()
        self.ranker = ranker or ComplexityRanker(self.scorer)
        self.selector = selector or ParetoSelector(self.scorer)
        self.refiner = refiner or RuleRefiner(self.generator, self.verifier)

    def run(
        self,
        examples: List[IOExample],
        max_iterations: int = 5,
        candidates_per_iteration: int = 5,
    ) -> SynthesisResult:
        """Run the full synthesis loop.

        Args:
            examples: I/O examples to synthesize rules for.
            max_iterations: Maximum number of iterations.
            candidates_per_iteration: Number of candidates per iteration.

        Returns:
            SynthesisResult with all iteration results and best rules.
        """
        result = SynthesisResult()
        all_scored_rules: List[ScoredRule] = []
        best_rule_so_far: Optional[CandidateRule] = None

        for iteration in range(max_iterations):
            iter_result = self.run_iteration(
                examples=examples,
                iteration=iteration,
                n_candidates=candidates_per_iteration,
                previous_best=best_rule_so_far,
            )
            result.iterations.append(iter_result)
            result.total_candidates += iter_result.candidates_generated

            all_scored_rules.extend(iter_result.scored_rules)

            # Update best rule
            if iter_result.best_rule is not None:
                if (
                    best_rule_so_far is None
                    or iter_result.best_accuracy > result.final_best_accuracy
                ):
                    best_rule_so_far = iter_result.best_rule
                    result.final_best_accuracy = iter_result.best_accuracy

            # Early termination if perfect accuracy
            if iter_result.best_accuracy >= 1.0:
                break

        result.total_iterations = len(result.iterations)

        # Final Pareto selection across all iterations
        if all_scored_rules:
            pareto_front = self.selector.compute_pareto_front(all_scored_rules)
            result.best_rules = pareto_front

        return result

    def run_iteration(
        self,
        examples: List[IOExample],
        iteration: int = 0,
        n_candidates: int = 5,
        previous_best: Optional[CandidateRule] = None,
    ) -> IterationResult:
        """Run a single synthesis iteration.

        Args:
            examples: I/O examples.
            iteration: Iteration number.
            n_candidates: Number of candidates to generate.
            previous_best: Best rule from previous iteration for refinement.

        Returns:
            IterationResult.
        """
        # 1. Generate candidates
        candidates = self.generator.generate(
            examples, n=n_candidates
        )

        # If we have a previous best, also generate refinements
        if previous_best is not None:
            vr = self.verifier.verify(previous_best, examples)
            if vr.failures:
                refined = self.refiner.refine(
                    previous_best, vr.failures, examples
                )
                candidates.extend(refined)

        # 2. Verify all candidates
        verification_results: Dict[str, VerificationResult] = {}
        for candidate in candidates:
            vr = self.verifier.verify(candidate, examples)
            verification_results[candidate.rule_id] = vr

        # 3. Select with Pareto
        scored_rules = self.selector.select(
            candidates, verification_results, examples
        )

        # 4. Find best
        best_accuracy = 0.0
        best_rule = None
        for sr in scored_rules:
            if sr.accuracy > best_accuracy:
                best_accuracy = sr.accuracy
                best_rule = sr.rule

        pareto_count = sum(1 for sr in scored_rules if sr.is_pareto_optimal)

        return IterationResult(
            iteration=iteration,
            candidates_generated=len(candidates),
            candidates_verified=len(verification_results),
            best_accuracy=best_accuracy,
            best_rule=best_rule,
            pareto_front_size=pareto_count,
            scored_rules=scored_rules,
        )
