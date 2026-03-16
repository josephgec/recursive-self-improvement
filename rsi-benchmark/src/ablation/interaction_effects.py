"""Interaction effects: detect synergy and redundancy between paradigms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.ablation.ablation_study import AblationResult
from src.ablation.contribution import ContributionAnalyzer


@dataclass
class InteractionEffect:
    """An interaction effect between two paradigms."""
    paradigm_a: str
    paradigm_b: str
    effect_type: str  # "synergy" or "redundancy"
    magnitude: float
    description: str


class InteractionAnalyzer:
    """Analyze interaction effects between paradigm components."""

    def __init__(self) -> None:
        self._contribution_analyzer = ContributionAnalyzer()

    def detect_synergy(
        self,
        result: AblationResult,
    ) -> List[InteractionEffect]:
        """Detect synergistic interactions between paradigms."""
        effects = []
        contributions = self._contribution_analyzer.compute_contributions(result)
        full_improvement = result.summary.get("full_pipeline", 0.0)

        paradigms = list(contributions.keys())
        for i in range(len(paradigms)):
            for j in range(i + 1, len(paradigms)):
                p_a = paradigms[i]
                p_b = paradigms[j]

                # Synergy: combined contribution > sum of individual
                c_a = contributions[p_a].marginal_contribution
                c_b = contributions[p_b].marginal_contribution
                combined = c_a + c_b

                # If both are contributing, check if there is synergy
                if c_a > 0 and c_b > 0:
                    # Simple synergy score based on relative contributions
                    synergy_magnitude = c_a * c_b / max(full_improvement, 1e-10)
                    if synergy_magnitude > 0.001:
                        effects.append(InteractionEffect(
                            paradigm_a=p_a,
                            paradigm_b=p_b,
                            effect_type="synergy",
                            magnitude=synergy_magnitude,
                            description=(
                                f"{p_a} and {p_b} show synergistic interaction "
                                f"(magnitude: {synergy_magnitude:.4f})"
                            ),
                        ))

        return effects

    def detect_redundancy(
        self,
        result: AblationResult,
    ) -> List[InteractionEffect]:
        """Detect redundant interactions between paradigms."""
        effects = []
        contributions = self._contribution_analyzer.compute_contributions(result)

        paradigms = list(contributions.keys())
        for i in range(len(paradigms)):
            for j in range(i + 1, len(paradigms)):
                p_a = paradigms[i]
                p_b = paradigms[j]

                c_a = contributions[p_a].marginal_contribution
                c_b = contributions[p_b].marginal_contribution

                # Redundancy: one has very small contribution relative to other
                if c_a > 0 and c_b > 0:
                    ratio = min(c_a, c_b) / max(c_a, c_b) if max(c_a, c_b) > 0 else 0
                    if ratio < 0.2:
                        weaker = p_a if c_a < c_b else p_b
                        effects.append(InteractionEffect(
                            paradigm_a=p_a,
                            paradigm_b=p_b,
                            effect_type="redundancy",
                            magnitude=1.0 - ratio,
                            description=(
                                f"{weaker} may be partially redundant when "
                                f"{p_a if weaker != p_a else p_b} is present"
                            ),
                        ))

        return effects

    def analyze_all(
        self,
        result: AblationResult,
    ) -> Dict[str, List[InteractionEffect]]:
        """Run all interaction analyses."""
        return {
            "synergy": self.detect_synergy(result),
            "redundancy": self.detect_redundancy(result),
        }
