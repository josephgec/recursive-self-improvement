"""Cross-suite interaction analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.suites.base import AblationSuiteResult


@dataclass
class InteractionReport:
    """Report on cross-paradigm interactions between suites."""

    suite_a: str
    suite_b: str
    synergy_score: float
    shared_patterns: List[str] = field(default_factory=list)
    divergent_patterns: List[str] = field(default_factory=list)
    summary: str = ""

    @property
    def has_synergy(self) -> bool:
        return self.synergy_score > 0.0


class CrossSuiteInteractionAnalyzer:
    """Analyze interactions between different ablation suites.

    Compares patterns across paradigms to identify shared and
    divergent component contributions.
    """

    def test_cross_paradigm_interactions(
        self,
        results: Dict[str, AblationSuiteResult],
    ) -> List[InteractionReport]:
        """Test for interactions between all pairs of suites."""
        reports = []
        suite_names = list(results.keys())

        for i in range(len(suite_names)):
            for j in range(i + 1, len(suite_names)):
                name_a = suite_names[i]
                name_b = suite_names[j]
                report = self._compare_suites(
                    name_a, results[name_a],
                    name_b, results[name_b],
                )
                reports.append(report)

        return reports

    def compute_synergy_score(
        self,
        result_a: AblationSuiteResult,
        result_b: AblationSuiteResult,
    ) -> float:
        """Compute a synergy score between two suites.

        Based on correlation of ablation effects: if removing similar
        components has similar effects across paradigms, synergy is high.
        """
        effects_a = self._compute_ablation_effects(result_a)
        effects_b = self._compute_ablation_effects(result_b)

        if not effects_a or not effects_b:
            return 0.0

        # Normalize effects to [0, 1] range
        max_a = max(abs(e) for e in effects_a.values()) if effects_a else 1.0
        max_b = max(abs(e) for e in effects_b.values()) if effects_b else 1.0

        if max_a == 0 or max_b == 0:
            return 0.0

        norm_a = {k: v / max_a for k, v in effects_a.items()}
        norm_b = {k: v / max_b for k, v in effects_b.items()}

        # Average absolute normalized effect as a proxy for synergy
        all_effects = list(norm_a.values()) + list(norm_b.values())
        if not all_effects:
            return 0.0

        return sum(abs(e) for e in all_effects) / len(all_effects)

    def _compare_suites(
        self,
        name_a: str,
        result_a: AblationSuiteResult,
        name_b: str,
        result_b: AblationSuiteResult,
    ) -> InteractionReport:
        """Compare two suites for shared patterns."""
        effects_a = self._compute_ablation_effects(result_a)
        effects_b = self._compute_ablation_effects(result_b)

        synergy = self.compute_synergy_score(result_a, result_b)

        shared = []
        divergent = []

        # Check if critical components show similar drop patterns
        threshold = 0.05
        for cond_a, drop_a in effects_a.items():
            for cond_b, drop_b in effects_b.items():
                if abs(drop_a) > threshold and abs(drop_b) > threshold:
                    if (drop_a > 0) == (drop_b > 0):
                        shared.append(
                            f"Both {cond_a} and {cond_b} show similar degradation"
                        )
                    else:
                        divergent.append(
                            f"{cond_a} and {cond_b} show opposite effects"
                        )

        # Limit to top 3 each
        shared = shared[:3]
        divergent = divergent[:3]

        summary = (
            f"Cross-paradigm analysis between {name_a} and {name_b}: "
            f"synergy={synergy:.3f}, {len(shared)} shared patterns, "
            f"{len(divergent)} divergent patterns."
        )

        return InteractionReport(
            suite_a=name_a,
            suite_b=name_b,
            synergy_score=synergy,
            shared_patterns=shared,
            divergent_patterns=divergent,
            summary=summary,
        )

    def _compute_ablation_effects(
        self, result: AblationSuiteResult
    ) -> Dict[str, float]:
        """Compute the effect of ablating each component.

        Effect = full_score - ablated_score (positive = component helps).
        """
        conditions = result.get_all_condition_names()
        full_name = None
        for c in conditions:
            if c == "full":
                full_name = c
                break

        if full_name is None:
            return {}

        full_score = result.get_mean_score(full_name)
        effects = {}
        for c in conditions:
            if c == full_name:
                continue
            effects[c] = full_score - result.get_mean_score(c)

        return effects
