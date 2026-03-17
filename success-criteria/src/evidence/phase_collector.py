"""Phase evidence collector — returns pre-built mock evidence."""

from __future__ import annotations

from typing import Any, Dict

from src.criteria.base import Evidence


class PhaseEvidenceCollector:
    """Collects evidence from all phases. Uses mock data for offline evaluation."""

    def __init__(self, evidence_paths: Dict[str, str] | None = None):
        self._evidence_paths = evidence_paths or {}

    def collect_all(self) -> Evidence:
        """Collect evidence from all phases and return an Evidence object."""
        return Evidence(
            phase_0=self._collect_phase("phase_0"),
            phase_1=self._collect_phase("phase_1"),
            phase_2=self._collect_phase("phase_2"),
            phase_3=self._collect_phase("phase_3"),
            phase_4=self._collect_phase("phase_4"),
            safety=self._collect_safety(),
            publications=self._collect_publications(),
            audit_trail=self._collect_audit_trail(),
        )

    def _collect_phase(self, phase: str) -> Dict[str, Any]:
        """Collect data for a single phase. Returns mock data."""
        mock_phases = self._get_mock_phases()
        return mock_phases.get(phase, {})

    def _collect_safety(self) -> Dict[str, Any]:
        """Collect safety evidence. Returns mock data."""
        return _build_mock_safety()

    def _collect_publications(self) -> list:
        """Collect publication evidence. Returns mock data."""
        return _build_mock_publications()

    def _collect_audit_trail(self) -> Dict[str, Any]:
        """Collect audit trail. Returns mock data."""
        return _build_mock_audit_trail()

    def _get_mock_phases(self) -> Dict[str, Dict[str, Any]]:
        """Build mock phase data with improvement curves and ablations."""
        return {
            "phase_0": {
                "score": 50.0,
                "collapse_score": 50.0,
                "ablations": {
                    "symcode": {"with": 52.0, "without": 45.0},
                    "godel": {"with": 51.0, "without": 48.0},
                    "soar": {"with": 53.0, "without": 46.0},
                    "rlm": {"with": 55.0, "without": 42.0},
                },
            },
            "phase_1": {
                "score": 55.0,
                "collapse_score": 49.0,
                "ablations": {
                    "symcode": {"with": 58.0, "without": 50.0},
                    "godel": {"with": 56.0, "without": 52.0},
                    "soar": {"with": 59.0, "without": 51.0},
                    "rlm": {"with": 62.0, "without": 48.0},
                },
            },
            "phase_2": {
                "score": 58.0,
                "collapse_score": 48.0,
                "ablations": {
                    "symcode": {"with": 62.0, "without": 53.0},
                    "godel": {"with": 60.0, "without": 55.0},
                    "soar": {"with": 63.0, "without": 54.0},
                    "rlm": {"with": 67.0, "without": 52.0},
                },
            },
            "phase_3": {
                "score": 62.0,
                "collapse_score": 47.0,
                "ablations": {
                    "symcode": {"with": 66.0, "without": 57.0},
                    "godel": {"with": 64.0, "without": 59.0},
                    "soar": {"with": 67.0, "without": 58.0},
                    "rlm": {"with": 72.0, "without": 56.0},
                },
            },
            "phase_4": {
                "score": 65.0,
                "collapse_score": 46.0,
                "ablations": {
                    "symcode": {"with": 70.0, "without": 61.0},
                    "godel": {"with": 68.0, "without": 63.0},
                    "soar": {"with": 71.0, "without": 62.0},
                    "rlm": {"with": 78.0, "without": 60.0},
                },
            },
        }


def _build_mock_safety() -> Dict[str, Any]:
    """Build mock safety evidence with GDI readings."""
    import hashlib

    readings = []
    for i in range(25):
        gdi = 0.20 + (i % 5) * 0.05  # Oscillates 0.20 to 0.40
        status = "yellow" if gdi > 0.30 else "green"
        readings.append({
            "timestamp": f"2025-{(i // 3) + 1:02d}-{(i % 28) + 1:02d}",
            "gdi": gdi,
            "status": status,
            "phase": f"phase_{min(i // 5, 4)}",
        })

    return {
        "gdi_readings": readings,
        "phases_monitored": [
            "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
        ],
    }


def _build_mock_publications() -> list:
    """Build mock publication list."""
    return [
        {
            "title": "Recursive Self-Improvement with Safety Guarantees",
            "venue": "NeurIPS",
            "status": "accepted",
            "year": 2025,
        },
        {
            "title": "GDI: A Guardrail Divergence Index for Safe AI",
            "venue": "ICML",
            "status": "accepted",
            "year": 2025,
        },
        {
            "title": "Paradigm Ablation in Self-Improving Systems",
            "venue": "ICLR",
            "status": "under_review",
            "year": 2026,
        },
    ]


def _build_mock_audit_trail() -> Dict[str, Any]:
    """Build mock audit trail with hash chain."""
    import hashlib

    mod_log = [
        {"timestamp": f"2025-{i+1:02d}-01", "action": f"modification_{i}",
         "description": f"Phase {i} code update"}
        for i in range(5)
    ]
    constraint_log = [
        {"timestamp": f"2025-{i+1:02d}-15", "constraint": f"C{i}",
         "status": "satisfied"}
        for i in range(5)
    ]
    gdi_log = [
        {"timestamp": f"2025-{i+1:02d}-10", "gdi": 0.25 + i * 0.03,
         "action": "monitored"}
        for i in range(5)
    ]
    interp_log = [
        {"timestamp": f"2025-{i+1:02d}-20",
         "interpretation": f"Model behavior analysis {i}"}
        for i in range(5)
    ]
    traces = [
        {"id": j, "reasoning": f"Trace {j}: decision rationale"}
        for j in range(25)
    ]

    # Build hash chain
    chain = []
    prev_hash = "genesis"
    for i in range(10):
        data = f"entry_{i}"
        h = hashlib.sha256(f"{prev_hash}{data}".encode("utf-8")).hexdigest()
        chain.append({"data": data, "hash": h})
        prev_hash = h

    return {
        "modification_log": mod_log,
        "constraint_log": constraint_log,
        "gdi_log": gdi_log,
        "interp_log": interp_log,
        "reasoning_traces": traces,
        "hash_chain": chain,
    }
