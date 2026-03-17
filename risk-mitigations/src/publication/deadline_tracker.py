"""Deadline tracker for publication timelines.

Tracks conference deadlines and provides severity assessments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# Built-in conference deadlines
CONFERENCE_DEADLINES = [
    {
        "name": "NeurIPS 2026",
        "abstract_date": "2026-05-15",
        "submission_date": "2026-05-22",
    },
    {
        "name": "ICML 2026",
        "abstract_date": "2026-01-30",
        "submission_date": "2026-02-06",
    },
    {
        "name": "ICLR 2027",
        "abstract_date": "2026-09-28",
        "submission_date": "2026-10-03",
    },
]


@dataclass
class DeadlineStatus:
    """Status of a deadline."""
    name: str
    abstract_date: str
    submission_date: str
    days_to_abstract: int
    days_to_submission: int
    severity: str  # "relaxed", "approaching", "urgent", "critical", "past"
    is_past: bool

    @property
    def is_urgent(self) -> bool:
        return self.severity in ("urgent", "critical")


class DeadlineTracker:
    """Tracks publication deadlines and provides severity assessments.

    Built-in conference deadlines with configurable additional deadlines.
    """

    def __init__(
        self,
        deadlines: Optional[List[Dict[str, str]]] = None,
        reference_date: Optional[str] = None,
    ):
        """
        Args:
            deadlines: List of deadline dicts with 'name', 'abstract_date', 'submission_date'.
            reference_date: Override current date for testing (ISO format YYYY-MM-DD).
        """
        self.deadlines = deadlines or CONFERENCE_DEADLINES
        self._reference_date = reference_date

    def _now(self) -> datetime:
        """Get current date (or reference date for testing)."""
        if self._reference_date:
            return datetime.strptime(self._reference_date, "%Y-%m-%d")
        return datetime.now()

    def check(self, status: Optional[Dict[str, Any]] = None) -> List[DeadlineStatus]:
        """Check all deadlines and return their statuses.

        Args:
            status: Optional project status dict (unused in basic check).

        Returns:
            List of DeadlineStatus for each deadline.
        """
        now = self._now()
        results = []

        for deadline in self.deadlines:
            abstract_dt = datetime.strptime(deadline["abstract_date"], "%Y-%m-%d")
            submission_dt = datetime.strptime(deadline["submission_date"], "%Y-%m-%d")

            days_to_abstract = (abstract_dt - now).days
            days_to_submission = (submission_dt - now).days

            severity = self._compute_severity(days_to_submission)
            is_past = days_to_submission < 0

            results.append(DeadlineStatus(
                name=deadline["name"],
                abstract_date=deadline["abstract_date"],
                submission_date=deadline["submission_date"],
                days_to_abstract=days_to_abstract,
                days_to_submission=days_to_submission,
                severity=severity,
                is_past=is_past,
            ))

        return results

    def next_deadline(self) -> Optional[DeadlineStatus]:
        """Return the next upcoming deadline.

        Returns:
            DeadlineStatus for the nearest future deadline, or None.
        """
        statuses = self.check()
        future = [s for s in statuses if not s.is_past]
        if not future:
            return None
        return min(future, key=lambda s: s.days_to_submission)

    def severity(self, deadline_name: str) -> str:
        """Get severity for a specific deadline.

        Args:
            deadline_name: Name of the deadline.

        Returns:
            Severity string.
        """
        statuses = self.check()
        for s in statuses:
            if s.name == deadline_name:
                return s.severity
        raise KeyError(f"Unknown deadline: {deadline_name}")

    def _compute_severity(self, days_remaining: int) -> str:
        """Compute severity based on days remaining."""
        if days_remaining < 0:
            return "past"
        elif days_remaining <= 7:
            return "critical"
        elif days_remaining <= 21:
            return "urgent"
        elif days_remaining <= 60:
            return "approaching"
        else:
            return "relaxed"
