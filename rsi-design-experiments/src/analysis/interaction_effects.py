"""Interaction effects analysis between experimental factors."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.analysis.anova import ANOVAAnalyzer, ANOVAResult


@dataclass
class InteractionResult:
    """Result of interaction analysis between two factors."""

    factor_a_name: str
    factor_b_name: str
    main_effect_a: ANOVAResult
    main_effect_b: ANOVAResult
    interaction_significant: bool
    interaction_strength: float
    cell_means: Dict[Tuple[str, str], float] = field(default_factory=dict)


class InteractionAnalyzer:
    """Analyzes interaction effects between experimental factors."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self._anova = ANOVAAnalyzer(significance_level)

    def two_way_anova(
        self,
        factor_a_results: Dict[str, List[float]],
        factor_b_results: Dict[str, List[float]],
        factor_a_name: str = "factor_a",
        factor_b_name: str = "factor_b",
    ) -> InteractionResult:
        """Perform a simplified two-way ANOVA analysis.

        Since we have separate experiments for each factor, we approximate
        the interaction by checking if the optimal level of factor A changes
        depending on factor B level, and vice versa.
        """
        main_a = self._anova.one_way_anova(factor_a_results)
        main_b = self._anova.one_way_anova(factor_b_results)

        # Approximate interaction by looking at whether the variance patterns
        # of the two factors are correlated
        interaction_strength = self._estimate_interaction(
            factor_a_results, factor_b_results
        )
        interaction_significant = interaction_strength > 0.1

        return InteractionResult(
            factor_a_name=factor_a_name,
            factor_b_name=factor_b_name,
            main_effect_a=main_a,
            main_effect_b=main_b,
            interaction_significant=interaction_significant,
            interaction_strength=interaction_strength,
        )

    def detect_interactions(
        self,
        all_results: Dict[str, Dict[str, List[float]]],
    ) -> List[InteractionResult]:
        """Detect interactions between all pairs of experimental factors.

        Args:
            all_results: mapping of experiment_name -> condition_results
        """
        experiment_names = list(all_results.keys())
        interactions = []

        for i in range(len(experiment_names)):
            for j in range(i + 1, len(experiment_names)):
                name_a = experiment_names[i]
                name_b = experiment_names[j]
                result = self.two_way_anova(
                    all_results[name_a],
                    all_results[name_b],
                    factor_a_name=name_a,
                    factor_b_name=name_b,
                )
                interactions.append(result)

        return interactions

    @staticmethod
    def _estimate_interaction(
        factor_a: Dict[str, List[float]],
        factor_b: Dict[str, List[float]],
    ) -> float:
        """Estimate interaction strength between two factors.

        Uses the coefficient of variation ratio as a proxy.
        """
        def _coefficient_of_variation(groups: Dict[str, List[float]]) -> float:
            means = []
            for vals in groups.values():
                if vals:
                    means.append(sum(vals) / len(vals))
            if not means or len(means) < 2:
                return 0.0
            grand_mean = sum(means) / len(means)
            if grand_mean == 0:
                return 0.0
            variance = sum((m - grand_mean) ** 2 for m in means) / len(means)
            import math
            return math.sqrt(variance) / abs(grand_mean)

        cv_a = _coefficient_of_variation(factor_a)
        cv_b = _coefficient_of_variation(factor_b)

        # Interaction strength approximated as product of individual effects
        # normalized by their sum
        if cv_a + cv_b == 0:
            return 0.0
        return 2.0 * cv_a * cv_b / (cv_a + cv_b)
