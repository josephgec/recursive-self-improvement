"""Readiness checker for publication.

Evaluates whether experiments and writing are ready for submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReadinessReport:
    """Report on publication readiness."""
    ready: bool
    overall_readiness: float  # 0-1
    experiments_readiness: float
    writing_readiness: float
    blockers: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    component_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def gap_to_ready(self) -> float:
        return max(0.0, 1.0 - self.overall_readiness)


class ReadinessChecker:
    """Checks readiness for publication.

    Evaluates experiments and writing completeness against thresholds.
    """

    def __init__(
        self,
        min_experiments_complete: float = 0.8,
        min_writing_complete: float = 0.6,
    ):
        self.min_experiments_complete = min_experiments_complete
        self.min_writing_complete = min_writing_complete

    def check(
        self,
        experiments: Dict[str, Any],
        writing: Dict[str, Any],
    ) -> ReadinessReport:
        """Check publication readiness.

        Args:
            experiments: Dict with experiment status. Keys:
                - 'completed': number of completed experiments
                - 'total': total planned experiments
                - 'results_quality': quality score 0-1
            writing: Dict with writing status. Keys:
                - 'sections_done': number of completed sections
                - 'total_sections': total sections planned
                - 'quality': quality score 0-1

        Returns:
            ReadinessReport with assessment.
        """
        # Experiment readiness
        exp_completed = experiments.get("completed", 0)
        exp_total = experiments.get("total", 1)
        exp_quality = experiments.get("results_quality", 0.0)
        exp_fraction = exp_completed / exp_total if exp_total > 0 else 0.0
        exp_readiness = exp_fraction * 0.7 + exp_quality * 0.3

        # Writing readiness
        sections_done = writing.get("sections_done", 0)
        total_sections = writing.get("total_sections", 1)
        writing_quality = writing.get("quality", 0.0)
        writing_fraction = sections_done / total_sections if total_sections > 0 else 0.0
        writing_readiness = writing_fraction * 0.6 + writing_quality * 0.4

        # Overall
        overall = exp_readiness * 0.6 + writing_readiness * 0.4

        # Blockers
        blockers = []
        if exp_readiness < self.min_experiments_complete:
            blockers.append(
                f"Experiments at {exp_readiness:.0%}, need {self.min_experiments_complete:.0%}"
            )
        if writing_readiness < self.min_writing_complete:
            blockers.append(
                f"Writing at {writing_readiness:.0%}, need {self.min_writing_complete:.0%}"
            )

        # Recommendations
        recommendations = []
        if exp_fraction < 1.0:
            recommendations.append(
                f"Complete {exp_total - exp_completed} remaining experiments"
            )
        if writing_fraction < 1.0:
            recommendations.append(
                f"Complete {total_sections - sections_done} remaining sections"
            )
        if exp_quality < 0.7:
            recommendations.append("Improve experiment result quality")
        if writing_quality < 0.7:
            recommendations.append("Improve writing quality")

        ready = len(blockers) == 0

        return ReadinessReport(
            ready=ready,
            overall_readiness=overall,
            experiments_readiness=exp_readiness,
            writing_readiness=writing_readiness,
            blockers=blockers,
            recommendations=recommendations,
            component_scores={
                "experiments_fraction": exp_fraction,
                "experiments_quality": exp_quality,
                "writing_fraction": writing_fraction,
                "writing_quality": writing_quality,
            },
        )
