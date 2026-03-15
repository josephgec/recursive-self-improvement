"""Complexity-based ranking of candidate rules using BDM and MDL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.bdm.scorer import BDMScorer, RuleScore
from src.synthesis.candidate_generator import CandidateRule, IOExample


@dataclass
class RankedRule:
    """A rule with its complexity ranking."""

    rule: CandidateRule
    rule_score: RuleScore
    rank: int = 0


class ComplexityRanker:
    """Ranks candidate rules by algorithmic complexity.

    Uses BDM scoring for program complexity and MDL for combined
    program + residual complexity.
    """

    def __init__(self, scorer: Optional[BDMScorer] = None) -> None:
        self.scorer = scorer or BDMScorer()

    def rank_by_bdm(
        self,
        rules: List[CandidateRule],
        examples: Optional[List[IOExample]] = None,
    ) -> List[RankedRule]:
        """Rank rules by BDM complexity (lower is simpler/better).

        Args:
            rules: List of candidate rules.
            examples: Optional I/O examples for accuracy computation.

        Returns:
            List of RankedRule sorted by BDM score (ascending).
        """
        ranked = []
        for rule in rules:
            inputs = [ex.input for ex in examples] if examples else []
            outputs = [ex.output for ex in examples] if examples else []

            rule_score = self.scorer.score_rule(rule.source_code, inputs, outputs)
            ranked.append(RankedRule(rule=rule, rule_score=rule_score))

        # Sort by BDM score (lower = simpler)
        ranked.sort(key=lambda r: r.rule_score.bdm_score)

        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    def rank_by_mdl(
        self,
        rules: List[CandidateRule],
        examples: Optional[List[IOExample]] = None,
    ) -> List[RankedRule]:
        """Rank rules by MDL: K(program) + K(residuals).

        The MDL principle favors rules that are simple AND explain the data well.

        Args:
            rules: List of candidate rules.
            examples: I/O examples for residual computation.

        Returns:
            List of RankedRule sorted by MDL score (ascending).
        """
        ranked = []
        for rule in rules:
            inputs = [ex.input for ex in examples] if examples else []
            outputs = [ex.output for ex in examples] if examples else []

            rule_score = self.scorer.score_rule(rule.source_code, inputs, outputs)
            ranked.append(RankedRule(rule=rule, rule_score=rule_score))

        # Sort by MDL score (lower = better balance of simplicity and fit)
        ranked.sort(key=lambda r: r.rule_score.mdl_score)

        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked
