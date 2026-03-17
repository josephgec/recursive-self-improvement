"""Modification quarantine for high-risk changes.

Quarantines modifications that exceed risk thresholds, requiring
explicit review and release before they can be deployed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class QuarantinedModification:
    """A modification held in quarantine."""
    modification_id: str
    reason: str
    blast_radius: float
    quarantined_at: str
    released: bool = False
    released_at: Optional[str] = None
    reviewer: Optional[str] = None


class ModificationQuarantine:
    """Quarantine system for high-risk modifications.

    Modifications exceeding the blast radius threshold are automatically
    quarantined and must be explicitly released after review.
    """

    def __init__(self, auto_quarantine_blast_radius: float = 0.7):
        self.auto_quarantine_threshold = auto_quarantine_blast_radius
        self._quarantined: Dict[str, QuarantinedModification] = {}
        self._released_history: List[QuarantinedModification] = []

    def should_quarantine(
        self,
        modification_id: str,
        blast_radius: float,
        risk_level: str = "low",
    ) -> bool:
        """Determine if a modification should be quarantined.

        Args:
            modification_id: Unique ID of the modification.
            blast_radius: Estimated blast radius (0-1).
            risk_level: Risk level string.

        Returns:
            True if the modification should be quarantined.
        """
        return (
            blast_radius >= self.auto_quarantine_threshold
            or risk_level in ("high", "critical")
        )

    def enter_quarantine(
        self,
        modification_id: str,
        reason: str,
        blast_radius: float,
    ) -> QuarantinedModification:
        """Place a modification in quarantine.

        Args:
            modification_id: Unique ID.
            reason: Why it was quarantined.
            blast_radius: The blast radius score.

        Returns:
            The QuarantinedModification record.
        """
        entry = QuarantinedModification(
            modification_id=modification_id,
            reason=reason,
            blast_radius=blast_radius,
            quarantined_at=datetime.now().isoformat(),
        )
        self._quarantined[modification_id] = entry
        return entry

    def check_quarantined(self, modification_id: str) -> bool:
        """Check if a modification is currently quarantined.

        Returns:
            True if the modification is in quarantine (not released).
        """
        entry = self._quarantined.get(modification_id)
        if entry is None:
            return False
        return not entry.released

    def release(
        self,
        modification_id: str,
        reviewer: str = "auto",
    ) -> QuarantinedModification:
        """Release a modification from quarantine.

        Args:
            modification_id: ID of the modification to release.
            reviewer: Who approved the release.

        Returns:
            The updated QuarantinedModification record.

        Raises:
            KeyError: If modification not found in quarantine.
        """
        entry = self._quarantined.get(modification_id)
        if entry is None:
            raise KeyError(f"Modification '{modification_id}' not in quarantine")

        entry.released = True
        entry.released_at = datetime.now().isoformat()
        entry.reviewer = reviewer
        self._released_history.append(entry)

        return entry

    def get_all_quarantined(self) -> List[QuarantinedModification]:
        """Return all currently quarantined (unreleased) modifications."""
        return [
            entry for entry in self._quarantined.values()
            if not entry.released
        ]

    def get_released_history(self) -> List[QuarantinedModification]:
        """Return history of released modifications."""
        return list(self._released_history)
