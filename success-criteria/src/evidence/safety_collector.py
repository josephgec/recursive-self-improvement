"""Safety evidence collector — collects from safety subsystems S.1-S.4."""

from __future__ import annotations

from typing import Any, Dict, List


class SafetyEvidenceCollector:
    """Collects safety evidence from subsystems S.1 through S.4."""

    def __init__(self) -> None:
        self._subsystems = {
            "S.1": "GDI Monitoring",
            "S.2": "Constraint Verification",
            "S.3": "Behavioral Bounds",
            "S.4": "Rollback Capability",
        }

    def collect(self) -> Dict[str, Any]:
        """Collect safety evidence from all subsystems."""
        evidence: Dict[str, Any] = {}
        for subsystem_id, subsystem_name in self._subsystems.items():
            evidence[subsystem_id] = self._collect_subsystem(
                subsystem_id, subsystem_name
            )
        return evidence

    def _collect_subsystem(
        self, subsystem_id: str, subsystem_name: str
    ) -> Dict[str, Any]:
        """Collect evidence from a single safety subsystem (mock)."""
        collectors = {
            "S.1": self._collect_gdi_monitoring,
            "S.2": self._collect_constraint_verification,
            "S.3": self._collect_behavioral_bounds,
            "S.4": self._collect_rollback_capability,
        }
        collector = collectors.get(subsystem_id, self._default_collector)
        return {
            "subsystem_id": subsystem_id,
            "subsystem_name": subsystem_name,
            "status": "operational",
            "data": collector(),
        }

    @staticmethod
    def _collect_gdi_monitoring() -> Dict[str, Any]:
        """Mock GDI monitoring data."""
        return {
            "total_readings": 25,
            "max_gdi": 0.40,
            "mean_gdi": 0.28,
            "alerts": 0,
            "coverage": "all_phases",
        }

    @staticmethod
    def _collect_constraint_verification() -> Dict[str, Any]:
        """Mock constraint verification data."""
        return {
            "constraints_checked": 15,
            "constraints_satisfied": 15,
            "violations": 0,
            "last_check": "2026-03-01",
        }

    @staticmethod
    def _collect_behavioral_bounds() -> Dict[str, Any]:
        """Mock behavioral bounds data."""
        return {
            "bounds_defined": 10,
            "bounds_respected": 10,
            "excursions": 0,
            "max_excursion_magnitude": 0.0,
        }

    @staticmethod
    def _collect_rollback_capability() -> Dict[str, Any]:
        """Mock rollback capability data."""
        return {
            "rollback_tested": True,
            "rollback_successful": True,
            "recovery_time_seconds": 12.5,
            "checkpoints_available": 5,
        }

    @staticmethod
    def _default_collector() -> Dict[str, Any]:
        """Default collector for unknown subsystems."""
        return {"status": "unknown"}

    def get_subsystem_names(self) -> Dict[str, str]:
        """Return mapping of subsystem IDs to names."""
        return dict(self._subsystems)

    def get_subsystem_ids(self) -> List[str]:
        """Return list of subsystem IDs."""
        return list(self._subsystems.keys())
