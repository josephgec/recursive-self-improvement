"""Test fixtures — mock evidence factories."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import pytest

from src.criteria.base import Evidence


def _build_hash_chain(n: int = 10) -> List[Dict[str, str]]:
    """Build a valid SHA-256 hash chain."""
    chain = []
    prev_hash = "genesis"
    for i in range(n):
        data = f"entry_{i}"
        h = hashlib.sha256(f"{prev_hash}{data}".encode("utf-8")).hexdigest()
        chain.append({"data": data, "hash": h})
        prev_hash = h
    return chain


def _build_broken_hash_chain(n: int = 10, break_at: int = 5) -> List[Dict[str, str]]:
    """Build a hash chain with a break at the given index."""
    chain = _build_hash_chain(n)
    if break_at < len(chain):
        chain[break_at]["hash"] = "0000_broken_hash_0000"
    return chain


def _build_phase_data(
    scores: List[float],
    collapse_scores: List[float],
    ablation_diffs: Dict[str, List[float]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Build phase data dictionaries.

    Args:
        scores: Improvement scores for phases 0-4.
        collapse_scores: Collapse baseline scores for phases 0-4.
        ablation_diffs: Dict mapping paradigm name to list of
            (with - without) differences per phase. The 'with' score
            is set to score + diff/2, 'without' to score - diff/2.
    """
    default_diffs = {
        "symcode": [7.0, 8.0, 9.0, 9.0, 9.0],
        "godel": [3.0, 4.0, 5.0, 5.0, 5.0],
        "soar": [7.0, 8.0, 9.0, 9.0, 9.0],
        "rlm": [13.0, 14.0, 15.0, 16.0, 18.0],
    }
    diffs = ablation_diffs or default_diffs

    phases = {}
    for i in range(5):
        phase_key = f"phase_{i}"
        ablations = {}
        for paradigm, paradigm_diffs in diffs.items():
            d = paradigm_diffs[i] if i < len(paradigm_diffs) else 0.0
            base = scores[i] if i < len(scores) else 50.0
            ablations[paradigm] = {
                "with": base + d / 2,
                "without": base - d / 2,
            }
        phases[phase_key] = {
            "score": scores[i] if i < len(scores) else 50.0,
            "collapse_score": (
                collapse_scores[i] if i < len(collapse_scores) else 50.0
            ),
            "ablations": ablations,
        }
    return phases


def build_passing_evidence() -> Evidence:
    """Build evidence where all 5 criteria pass.

    - Increasing scores: 50, 55, 58, 62, 65 (gain=15pp)
    - Collapse: 50, 49, 48, 47, 46 (divergence=65-46=19pp)
    - All ablation diffs exceed thresholds
    - Max GDI = 0.40, no long yellow streaks
    - 2 accepted tier-1 publications
    - Complete audit trail with 25 traces and valid hash chain
    """
    scores = [50.0, 55.0, 58.0, 62.0, 65.0]
    collapse = [50.0, 49.0, 48.0, 47.0, 46.0]

    phases = _build_phase_data(scores, collapse)

    # Safety data
    gdi_readings = []
    for i in range(25):
        gdi = 0.20 + (i % 5) * 0.05
        status = "yellow" if gdi > 0.30 else "green"
        gdi_readings.append({
            "timestamp": f"2025-{(i // 3) + 1:02d}-{(i % 28) + 1:02d}",
            "gdi": gdi,
            "status": status,
            "phase": f"phase_{min(i // 5, 4)}",
        })

    safety = {
        "gdi_readings": gdi_readings,
        "phases_monitored": [
            "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
        ],
    }

    publications = [
        {
            "title": "Paper A",
            "venue": "NeurIPS",
            "status": "accepted",
            "year": 2025,
        },
        {
            "title": "Paper B",
            "venue": "ICML",
            "status": "accepted",
            "year": 2025,
        },
        {
            "title": "Paper C",
            "venue": "ICLR",
            "status": "under_review",
            "year": 2026,
        },
    ]

    audit_trail = {
        "modification_log": [
            {"timestamp": f"2025-{i+1:02d}-01", "action": f"mod_{i}"}
            for i in range(5)
        ],
        "constraint_log": [
            {"timestamp": f"2025-{i+1:02d}-15", "constraint": f"C{i}"}
            for i in range(5)
        ],
        "gdi_log": [
            {"timestamp": f"2025-{i+1:02d}-10", "gdi": 0.25 + i * 0.03}
            for i in range(5)
        ],
        "interp_log": [
            {"timestamp": f"2025-{i+1:02d}-20", "interp": f"I{i}"}
            for i in range(5)
        ],
        "reasoning_traces": [
            {"id": j, "reasoning": f"Trace {j}"} for j in range(25)
        ],
        "hash_chain": _build_hash_chain(10),
    }

    return Evidence(
        phase_0=phases["phase_0"],
        phase_1=phases["phase_1"],
        phase_2=phases["phase_2"],
        phase_3=phases["phase_3"],
        phase_4=phases["phase_4"],
        safety=safety,
        publications=publications,
        audit_trail=audit_trail,
    )


def build_partial_evidence() -> Evidence:
    """Build evidence where 3 of 5 criteria pass.

    Passes: Sustained Improvement, GDI Bounds, Auditability
    Fails: Paradigm Improvement (symcode diff too low), Publication (only workshops)
    """
    scores = [50.0, 55.0, 58.0, 62.0, 65.0]
    collapse = [50.0, 49.0, 48.0, 47.0, 46.0]

    # Make symcode diffs too small (< 5pp threshold)
    ablation_diffs = {
        "symcode": [2.0, 2.5, 3.0, 2.5, 3.0],  # ~2.6pp avg, fails 5pp
        "godel": [3.0, 4.0, 5.0, 5.0, 5.0],     # passes
        "soar": [7.0, 8.0, 9.0, 9.0, 9.0],      # passes
        "rlm": [13.0, 14.0, 15.0, 16.0, 18.0],  # passes
    }

    phases = _build_phase_data(scores, collapse, ablation_diffs)

    safety = {
        "gdi_readings": [
            {"timestamp": f"2025-{i+1:02d}", "gdi": 0.25, "status": "green",
             "phase": f"phase_{min(i // 5, 4)}"}
            for i in range(25)
        ],
        "phases_monitored": [
            "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
        ],
    }

    # Only workshop publications — fails tier requirement
    publications = [
        {
            "title": "Workshop Paper A",
            "venue": "SafeAI Workshop",
            "status": "accepted",
            "year": 2025,
        },
        {
            "title": "Workshop Paper B",
            "venue": "AI Safety Workshop",
            "status": "accepted",
            "year": 2025,
        },
    ]

    audit_trail = {
        "modification_log": [{"action": f"mod_{i}"} for i in range(5)],
        "constraint_log": [{"constraint": f"C{i}"} for i in range(5)],
        "gdi_log": [{"gdi": 0.25} for _ in range(5)],
        "interp_log": [{"interp": f"I{i}"} for i in range(5)],
        "reasoning_traces": [{"id": j} for j in range(25)],
        "hash_chain": _build_hash_chain(10),
    }

    return Evidence(
        phase_0=phases["phase_0"],
        phase_1=phases["phase_1"],
        phase_2=phases["phase_2"],
        phase_3=phases["phase_3"],
        phase_4=phases["phase_4"],
        safety=safety,
        publications=publications,
        audit_trail=audit_trail,
    )


def build_failing_evidence() -> Evidence:
    """Build evidence where only 1 criterion passes.

    Passes: GDI Bounds only
    Fails: Sustained (flat), Paradigm (all diffs too low),
           Publication (0 accepted), Auditability (missing logs, broken chain)
    """
    # Flat scores — no improvement
    scores = [50.0, 50.5, 50.0, 50.5, 51.0]
    collapse = [50.0, 50.0, 50.0, 50.0, 50.0]

    # All ablation diffs too small
    ablation_diffs = {
        "symcode": [1.0, 1.0, 1.0, 1.0, 1.0],
        "godel": [0.5, 0.5, 0.5, 0.5, 0.5],
        "soar": [1.0, 1.0, 1.0, 1.0, 1.0],
        "rlm": [2.0, 2.0, 2.0, 2.0, 2.0],
    }

    phases = _build_phase_data(scores, collapse, ablation_diffs)

    # GDI is fine (this criterion passes)
    safety = {
        "gdi_readings": [
            {"timestamp": f"2025-{i+1:02d}", "gdi": 0.20, "status": "green",
             "phase": f"phase_{min(i // 5, 4)}"}
            for i in range(25)
        ],
        "phases_monitored": [
            "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
        ],
    }

    # No accepted publications
    publications = [
        {
            "title": "Rejected Paper",
            "venue": "NeurIPS",
            "status": "rejected",
            "year": 2025,
        },
    ]

    # Missing logs and broken hash chain
    audit_trail = {
        "modification_log": [],  # empty — fails
        "constraint_log": [],    # empty — fails
        # gdi_log missing entirely
        # interp_log missing entirely
        "reasoning_traces": [{"id": j} for j in range(5)],  # only 5 — fails
        "hash_chain": _build_broken_hash_chain(10, break_at=3),
    }

    return Evidence(
        phase_0=phases["phase_0"],
        phase_1=phases["phase_1"],
        phase_2=phases["phase_2"],
        phase_3=phases["phase_3"],
        phase_4=phases["phase_4"],
        safety=safety,
        publications=publications,
        audit_trail=audit_trail,
    )


# Pytest fixtures

@pytest.fixture
def passing_evidence() -> Evidence:
    """Evidence where all 5 criteria pass."""
    return build_passing_evidence()


@pytest.fixture
def partial_evidence() -> Evidence:
    """Evidence where 3 of 5 criteria pass."""
    return build_partial_evidence()


@pytest.fixture
def failing_evidence() -> Evidence:
    """Evidence where only 1 criterion passes."""
    return build_failing_evidence()
