"""Fallback planner for publication deadlines.

Plans fallback strategies when full paper may not be ready in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FallbackPlan:
    """A fallback plan for publication."""
    target_venue: str
    strategy: str  # "full_submission", "workshop", "arxiv_preprint", "defer"
    required_completions: List[str]
    estimated_days_needed: int
    feasible: bool
    priority: int
    notes: str = ""

    @property
    def is_deferred(self) -> bool:
        return self.strategy == "defer"


class FallbackPlanner:
    """Plans fallback strategies for publication deadlines.

    Given project readiness and deadlines, generates ordered fallback plans.
    """

    def plan(
        self,
        readiness: Dict[str, float],
        deadlines: List[Dict[str, Any]],
    ) -> List[FallbackPlan]:
        """Generate fallback plans.

        Args:
            readiness: Dict with keys like 'experiments', 'writing', 'figures'.
                Values are completion fractions (0-1).
            deadlines: List of deadline dicts with 'name', 'days_to_submission'.

        Returns:
            List of FallbackPlan options, ordered by priority.
        """
        plans = []

        for deadline in deadlines:
            name = deadline.get("name", "Unknown")
            days = deadline.get("days_to_submission", 0)

            if days < 0:
                continue  # Skip past deadlines

            exp_ready = readiness.get("experiments", 0.0)
            writing_ready = readiness.get("writing", 0.0)
            figures_ready = readiness.get("figures", 0.0)
            overall = (exp_ready + writing_ready + figures_ready) / 3.0

            if overall >= 0.8 and days >= 7:
                plans.append(FallbackPlan(
                    target_venue=name,
                    strategy="full_submission",
                    required_completions=self._remaining_items(readiness),
                    estimated_days_needed=max(7, int((1.0 - overall) * 30)),
                    feasible=True,
                    priority=1,
                    notes="Full paper submission is feasible",
                ))
            elif exp_ready >= 0.6 and days >= 3:
                plans.append(FallbackPlan(
                    target_venue=name,
                    strategy="workshop",
                    required_completions=["finalize key results", "write short paper"],
                    estimated_days_needed=5,
                    feasible=True,
                    priority=2,
                    notes="Submit as workshop paper with preliminary results",
                ))
            elif exp_ready >= 0.4:
                plans.append(FallbackPlan(
                    target_venue=name,
                    strategy="arxiv_preprint",
                    required_completions=["write up current results", "basic analysis"],
                    estimated_days_needed=10,
                    feasible=True,
                    priority=3,
                    notes="Post as arXiv preprint, submit to later venue",
                ))
            else:
                plans.append(FallbackPlan(
                    target_venue=name,
                    strategy="defer",
                    required_completions=["complete experiments", "write paper"],
                    estimated_days_needed=60,
                    feasible=False,
                    priority=4,
                    notes="Defer to a later venue",
                ))

        return sorted(plans, key=lambda p: p.priority)

    def minimum_viable_submission(
        self,
        readiness: Dict[str, float],
    ) -> Dict[str, Any]:
        """Determine the minimum viable submission given current readiness.

        Args:
            readiness: Completion fractions for each component.

        Returns:
            Dict describing the minimum viable submission.
        """
        exp_ready = readiness.get("experiments", 0.0)
        writing_ready = readiness.get("writing", 0.0)

        if exp_ready >= 0.8 and writing_ready >= 0.6:
            return {
                "type": "full_paper",
                "viable": True,
                "gaps": self._remaining_items(readiness),
                "estimated_effort_days": 7,
            }
        elif exp_ready >= 0.5:
            return {
                "type": "workshop_paper",
                "viable": True,
                "gaps": ["complete writing", "subset of experiments"],
                "estimated_effort_days": 5,
            }
        elif exp_ready >= 0.3:
            return {
                "type": "extended_abstract",
                "viable": True,
                "gaps": ["preliminary results only"],
                "estimated_effort_days": 3,
            }
        else:
            return {
                "type": "none",
                "viable": False,
                "gaps": ["insufficient experimental results"],
                "estimated_effort_days": 30,
            }

    def _remaining_items(self, readiness: Dict[str, float]) -> List[str]:
        """List remaining items based on readiness."""
        items = []
        if readiness.get("experiments", 0.0) < 1.0:
            items.append("complete remaining experiments")
        if readiness.get("writing", 0.0) < 1.0:
            items.append("finish writing")
        if readiness.get("figures", 0.0) < 1.0:
            items.append("finalize figures")
        if readiness.get("references", 0.0) < 1.0:
            items.append("complete references")
        return items
