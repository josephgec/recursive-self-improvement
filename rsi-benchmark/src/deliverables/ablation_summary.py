"""Ablation summary deliverable."""

from __future__ import annotations

from typing import Any, Dict, List

from src.ablation.ablation_study import AblationResult
from src.ablation.contribution import ContributionAnalyzer


class AblationSummary:
    """Generate ablation study summary."""

    def generate(self, result: AblationResult) -> Dict[str, Any]:
        analyzer = ContributionAnalyzer()
        ranked = analyzer.rank_paradigms(result)
        synergy = analyzer.compute_synergy(result)

        summary: Dict[str, Any] = {
            "conditions": result.conditions,
            "benchmarks": result.benchmarks,
            "improvement_by_condition": dict(result.summary),
            "paradigm_ranking": [
                {
                    "rank": c.rank,
                    "paradigm": c.paradigm,
                    "marginal_contribution": round(c.marginal_contribution, 4),
                    "relative_contribution": round(c.relative_contribution, 4),
                }
                for c in ranked
            ],
            "synergy_score": round(synergy, 4),
        }
        return summary

    def to_markdown(self, result: AblationResult) -> str:
        data = self.generate(result)
        lines = [
            "# Ablation Study Summary",
            "",
            "## Conditions Tested",
        ]
        for cond in data["conditions"]:
            imp = data["improvement_by_condition"].get(cond, 0)
            lines.append(f"- {cond}: improvement = {imp:.4f}")
        lines.append("")
        lines.append("## Paradigm Ranking")
        for p in data["paradigm_ranking"]:
            lines.append(
                f"{p['rank']}. {p['paradigm']} "
                f"(marginal: {p['marginal_contribution']:.4f}, "
                f"relative: {p['relative_contribution']:.4f})"
            )
        lines.append("")
        lines.append(f"## Synergy Score: {data['synergy_score']:.4f}")
        lines.append("")
        return "\n".join(lines)
