"""Contribution analyzer: compute paradigm contributions from ablation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.ablation.ablation_study import AblationResult


@dataclass
class ParadigmContribution:
    """Contribution of a single paradigm."""
    paradigm: str
    marginal_contribution: float  # Improvement drop when removed
    relative_contribution: float  # Fraction of total improvement
    rank: int = 0


class ContributionAnalyzer:
    """Analyze paradigm contributions from ablation results."""

    def compute_contributions(
        self,
        result: AblationResult,
    ) -> Dict[str, ParadigmContribution]:
        """Compute marginal contribution of each paradigm."""
        full_improvement = result.summary.get("full_pipeline", 0.0)
        contributions: Dict[str, ParadigmContribution] = {}

        paradigm_map = {
            "soar": "no_soar",
            "ctm": "no_ctm",
            "godel": "no_godel",
            "rlm": "no_rlm",
        }

        for paradigm, ablation_name in paradigm_map.items():
            ablation_improvement = result.summary.get(ablation_name, 0.0)
            marginal = full_improvement - ablation_improvement
            relative = marginal / full_improvement if full_improvement != 0 else 0.0

            contributions[paradigm] = ParadigmContribution(
                paradigm=paradigm,
                marginal_contribution=marginal,
                relative_contribution=relative,
            )

        return contributions

    def compute_synergy(
        self,
        result: AblationResult,
    ) -> float:
        """Compute synergy: how much the combined system exceeds sum of parts.

        Synergy > 0 means components work better together than alone.
        """
        full_improvement = result.summary.get("full_pipeline", 0.0)
        soar_only = result.summary.get("soar_only", 0.0)
        naive = result.summary.get("naive_self_train", 0.0)

        # Sum of individual marginal contributions
        contributions = self.compute_contributions(result)
        sum_marginals = sum(c.marginal_contribution for c in contributions.values())

        # Synergy = full - sum_marginals (if positive, synergistic)
        if full_improvement == 0:
            return 0.0
        return full_improvement - sum_marginals

    def rank_paradigms(
        self,
        result: AblationResult,
    ) -> List[ParadigmContribution]:
        """Rank paradigms by their contribution."""
        contributions = self.compute_contributions(result)
        ranked = sorted(
            contributions.values(),
            key=lambda c: c.marginal_contribution,
            reverse=True,
        )
        for i, c in enumerate(ranked):
            c.rank = i + 1
        return ranked

    def plot_contribution_waterfall(
        self,
        result: AblationResult,
    ) -> Dict[str, Any]:
        """Generate waterfall plot data for paradigm contributions."""
        ranked = self.rank_paradigms(result)
        full_improvement = result.summary.get("full_pipeline", 0.0)
        naive = result.summary.get("naive_self_train", 0.0)

        labels = ["naive_baseline"]
        values = [naive]
        cumulative = [naive]

        running = naive
        for c in ranked:
            labels.append(c.paradigm)
            values.append(c.marginal_contribution)
            running += c.marginal_contribution
            cumulative.append(running)

        labels.append("full_pipeline")
        values.append(0)
        cumulative.append(full_improvement)

        return {
            "labels": labels,
            "values": values,
            "cumulative": cumulative,
            "full_improvement": full_improvement,
        }

    def plot_ablation_curves(
        self,
        result: AblationResult,
    ) -> Dict[str, Any]:
        """Generate plot data for ablation accuracy curves."""
        plot_data: Dict[str, Any] = {}
        for condition, bm_runs in result.runs.items():
            # Average across benchmarks
            all_accuracies: List[List[float]] = []
            for run in bm_runs.values():
                all_accuracies.append(run.accuracies)

            if all_accuracies:
                n_iters = len(all_accuracies[0])
                avg_curve = []
                for i in range(n_iters):
                    avg = sum(acc[i] for acc in all_accuracies if i < len(acc)) / len(
                        all_accuracies
                    )
                    avg_curve.append(avg)
                plot_data[condition] = avg_curve

        return plot_data
