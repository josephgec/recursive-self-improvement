"""Pareto-optimal selection of rules balancing accuracy and complexity."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.bdm.scorer import BDMScorer
from src.synthesis.candidate_generator import CandidateRule, IOExample
from src.synthesis.empirical_verifier import VerificationResult


@dataclass
class ScoredRule:
    """A rule scored on both accuracy and complexity dimensions."""

    rule: CandidateRule
    accuracy: float
    bdm_complexity: float
    mdl_score: float
    is_pareto_optimal: bool = False
    pareto_rank: int = 0

    @property
    def fitness(self) -> float:
        """Combined fitness metric."""
        if self.accuracy <= 0:
            return float("inf")
        return self.bdm_complexity / self.accuracy


class ParetoSelector:
    """Selects rules on the Pareto front of accuracy vs. complexity.

    A rule is Pareto-optimal if no other rule is both more accurate
    AND less complex.
    """

    def __init__(self, scorer: Optional[BDMScorer] = None) -> None:
        self.scorer = scorer or BDMScorer()

    def select(
        self,
        candidates: List[CandidateRule],
        verification_results: Dict[str, VerificationResult],
        examples: Optional[List[IOExample]] = None,
    ) -> List[ScoredRule]:
        """Score and select Pareto-optimal rules.

        Args:
            candidates: List of candidate rules.
            verification_results: Mapping from rule_id to VerificationResult.
            examples: Optional I/O examples for scoring.

        Returns:
            List of ScoredRule, with is_pareto_optimal set for Pareto front rules.
        """
        scored_rules = []

        for rule in candidates:
            vr = verification_results.get(rule.rule_id)
            accuracy = vr.accuracy if vr else 0.0

            inputs = [ex.input for ex in examples] if examples else []
            outputs = [ex.output for ex in examples] if examples else []

            rule_score = self.scorer.score_rule(rule.source_code, inputs, outputs)

            scored_rules.append(
                ScoredRule(
                    rule=rule,
                    accuracy=accuracy,
                    bdm_complexity=rule_score.bdm_score,
                    mdl_score=rule_score.mdl_score,
                )
            )

        # Compute Pareto front
        pareto_front = self.compute_pareto_front(scored_rules)

        for sr in scored_rules:
            if sr in pareto_front:
                sr.is_pareto_optimal = True

        # Assign Pareto ranks
        self._assign_pareto_ranks(scored_rules)

        return scored_rules

    def compute_pareto_front(
        self, scored_rules: List[ScoredRule]
    ) -> List[ScoredRule]:
        """Compute the Pareto front: rules not dominated by any other.

        A rule A dominates rule B if A has higher accuracy AND lower complexity.
        The Pareto front consists of all non-dominated rules.

        Args:
            scored_rules: List of scored rules.

        Returns:
            List of Pareto-optimal rules.
        """
        if not scored_rules:
            return []

        pareto = []
        for candidate in scored_rules:
            dominated = False
            for other in scored_rules:
                if other is candidate:
                    continue
                # other dominates candidate if:
                # other.accuracy >= candidate.accuracy AND other.complexity <= candidate.complexity
                # with at least one strict inequality
                if (
                    other.accuracy >= candidate.accuracy
                    and other.bdm_complexity <= candidate.bdm_complexity
                    and (
                        other.accuracy > candidate.accuracy
                        or other.bdm_complexity < candidate.bdm_complexity
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)

        return pareto

    def _assign_pareto_ranks(self, scored_rules: List[ScoredRule]) -> None:
        """Assign Pareto ranks (1 = on the front, 2 = next layer, etc.)."""
        remaining = list(scored_rules)
        rank = 1

        while remaining:
            front = self.compute_pareto_front(remaining)
            if not front:
                break
            for sr in front:
                sr.pareto_rank = rank
            remaining = [sr for sr in remaining if sr not in front]
            rank += 1

    def plot_pareto_front(
        self,
        scored_rules: List[ScoredRule],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot the Pareto front (accuracy vs. complexity).

        Args:
            scored_rules: List of scored rules.
            output_path: Path to save the plot. If None, displays interactively.

        Returns:
            Path to saved plot, or None.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            accuracies = [sr.accuracy for sr in scored_rules]
            complexities = [sr.bdm_complexity for sr in scored_rules]
            pareto_mask = [sr.is_pareto_optimal for sr in scored_rules]

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # Non-Pareto points
            non_pareto_acc = [a for a, p in zip(accuracies, pareto_mask) if not p]
            non_pareto_comp = [c for c, p in zip(complexities, pareto_mask) if not p]
            ax.scatter(
                non_pareto_comp, non_pareto_acc,
                c="gray", alpha=0.5, label="Non-Pareto"
            )

            # Pareto front points
            pareto_acc = [a for a, p in zip(accuracies, pareto_mask) if p]
            pareto_comp = [c for c, p in zip(complexities, pareto_mask) if p]
            ax.scatter(
                pareto_comp, pareto_acc,
                c="red", s=100, zorder=5, label="Pareto Front"
            )

            # Connect Pareto front
            if pareto_comp:
                paired = sorted(zip(pareto_comp, pareto_acc))
                ax.plot(
                    [p[0] for p in paired],
                    [p[1] for p in paired],
                    "r--", alpha=0.5,
                )

            ax.set_xlabel("BDM Complexity")
            ax.set_ylabel("Accuracy")
            ax.set_title("Pareto Front: Accuracy vs. Complexity")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                return output_path
            else:
                plt.close(fig)
                return None
        except ImportError:
            return None
