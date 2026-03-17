"""Incident manager for risk events.

Manages the lifecycle of risk incidents: creation, status updates, resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


INCIDENT_STATUSES = ("open", "investigating", "mitigating", "resolved", "closed")
INCIDENT_SEVERITIES = ("low", "medium", "high", "critical")


@dataclass
class Incident:
    """A risk incident with full lifecycle tracking."""
    incident_id: str
    title: str
    description: str
    severity: str
    domain: str
    status: str = "open"
    created_at: str = ""
    updated_at: str = ""
    resolved_at: Optional[str] = None
    resolution: Optional[str] = None
    updates: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.status not in ("resolved", "closed")

    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"


class IncidentManager:
    """Manages the lifecycle of risk incidents.

    Supports creation, status updates, resolution, and history queries.
    """

    def __init__(self):
        self._incidents: Dict[str, Incident] = {}
        self._counter = 0

    def create_incident(
        self,
        title: str,
        description: str,
        severity: str,
        domain: str,
    ) -> Incident:
        """Create a new incident.

        Args:
            title: Short title.
            description: Detailed description.
            severity: One of 'low', 'medium', 'high', 'critical'.
            domain: Risk domain (e.g., 'collapse', 'cost').

        Returns:
            The created Incident.
        """
        if severity not in INCIDENT_SEVERITIES:
            raise ValueError(f"Invalid severity: {severity}. Must be one of {INCIDENT_SEVERITIES}")

        self._counter += 1
        incident_id = f"INC-{self._counter:04d}"
        now = datetime.now().isoformat()

        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            domain=domain,
            status="open",
            created_at=now,
            updated_at=now,
        )

        self._incidents[incident_id] = incident
        return incident

    def update_status(
        self,
        incident_id: str,
        new_status: str,
        note: str = "",
    ) -> Incident:
        """Update an incident's status.

        Args:
            incident_id: ID of the incident.
            new_status: New status value.
            note: Optional note about the update.

        Returns:
            The updated Incident.

        Raises:
            KeyError: If incident not found.
            ValueError: If invalid status.
        """
        if incident_id not in self._incidents:
            raise KeyError(f"Unknown incident: {incident_id}")
        if new_status not in INCIDENT_STATUSES:
            raise ValueError(f"Invalid status: {new_status}. Must be one of {INCIDENT_STATUSES}")

        incident = self._incidents[incident_id]
        now = datetime.now().isoformat()

        incident.updates.append({
            "old_status": incident.status,
            "new_status": new_status,
            "note": note,
            "timestamp": now,
        })

        incident.status = new_status
        incident.updated_at = now

        return incident

    def resolve(
        self,
        incident_id: str,
        resolution: str,
    ) -> Incident:
        """Resolve an incident.

        Args:
            incident_id: ID of the incident.
            resolution: Description of the resolution.

        Returns:
            The resolved Incident.
        """
        incident = self.update_status(incident_id, "resolved", f"Resolved: {resolution}")
        incident.resolved_at = datetime.now().isoformat()
        incident.resolution = resolution
        return incident

    def get_open(self) -> List[Incident]:
        """Return all open (unresolved) incidents."""
        return [i for i in self._incidents.values() if i.is_open]

    def get_history(self) -> List[Incident]:
        """Return all incidents (open and closed)."""
        return list(self._incidents.values())

    def get_incident(self, incident_id: str) -> Incident:
        """Get a specific incident.

        Raises:
            KeyError: If not found.
        """
        if incident_id not in self._incidents:
            raise KeyError(f"Unknown incident: {incident_id}")
        return self._incidents[incident_id]

    def get_by_domain(self, domain: str) -> List[Incident]:
        """Return all incidents for a domain."""
        return [i for i in self._incidents.values() if i.domain == domain]

    def get_by_severity(self, severity: str) -> List[Incident]:
        """Return all incidents of a given severity."""
        return [i for i in self._incidents.values() if i.severity == severity]

    @property
    def open_count(self) -> int:
        return len(self.get_open())

    @property
    def total_count(self) -> int:
        return len(self._incidents)
