"""Blast radius estimation for self-modifications.

Estimates the scope of impact of a proposed code change, considering
code change magnitude, function importance, and downstream dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BlastRadiusEstimate:
    """Estimate of a modification's blast radius."""
    code_change_magnitude: float  # 0-1, fraction of codebase affected
    function_importance: float  # 0-1, criticality of affected functions
    downstream_impact: float  # 0-1, impact on downstream components
    overall_risk: float  # 0-1, combined risk score
    affected_modules: List[str] = field(default_factory=list)
    risk_level: str = "low"  # "low", "medium", "high", "critical"

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level in ("high", "critical")


# Mock module dependency graph
MODULE_DEPENDENCIES = {
    "core": ["training", "inference", "evaluation"],
    "training": ["data_pipeline", "optimizer"],
    "inference": ["decoder", "tokenizer"],
    "evaluation": ["metrics", "benchmarks"],
    "data_pipeline": ["tokenizer"],
    "optimizer": [],
    "decoder": [],
    "tokenizer": [],
    "metrics": [],
    "benchmarks": ["metrics"],
}

# Mock function importance scores
FUNCTION_IMPORTANCE = {
    "core": 1.0,
    "training": 0.9,
    "inference": 0.9,
    "evaluation": 0.7,
    "data_pipeline": 0.6,
    "optimizer": 0.8,
    "decoder": 0.7,
    "tokenizer": 0.5,
    "metrics": 0.4,
    "benchmarks": 0.4,
}


class BlastRadiusEstimator:
    """Estimates the blast radius of a proposed modification.

    Considers:
    - Code change magnitude (how much code changes)
    - Function importance (how critical the changed functions are)
    - Downstream impact (what depends on the changed code)
    """

    def __init__(
        self,
        dependencies: Optional[Dict[str, List[str]]] = None,
        importance: Optional[Dict[str, float]] = None,
    ):
        self.dependencies = dependencies or MODULE_DEPENDENCIES
        self.importance = importance or FUNCTION_IMPORTANCE
        self._total_modules = len(self.dependencies)

    def estimate(self, modification: Dict[str, Any]) -> BlastRadiusEstimate:
        """Estimate the blast radius of a modification.

        Args:
            modification: Dict with:
                - 'affected_modules': List of directly modified modules
                - 'lines_changed': Number of lines changed (optional)
                - 'total_lines': Total lines in codebase (optional)

        Returns:
            BlastRadiusEstimate with risk assessment.
        """
        affected = modification.get("affected_modules", [])
        lines_changed = modification.get("lines_changed", 0)
        total_lines = modification.get("total_lines", 1000)

        # Code change magnitude
        if total_lines > 0:
            code_magnitude = min(lines_changed / total_lines, 1.0)
        else:
            code_magnitude = 0.0

        # Function importance (max importance of affected modules)
        if affected:
            func_importance = max(
                self.importance.get(m, 0.5) for m in affected
            )
        else:
            func_importance = 0.0

        # Downstream impact - find all transitively affected modules
        all_affected = set(affected)
        downstream = self._get_downstream(affected)
        all_affected.update(downstream)

        if self._total_modules > 0:
            downstream_impact = len(all_affected) / self._total_modules
        else:
            downstream_impact = 0.0

        # Overall risk is weighted combination
        overall_risk = (
            0.3 * code_magnitude
            + 0.3 * func_importance
            + 0.4 * downstream_impact
        )
        overall_risk = min(overall_risk, 1.0)

        # Risk level
        if overall_risk >= 0.7:
            risk_level = "critical"
        elif overall_risk >= 0.5:
            risk_level = "high"
        elif overall_risk >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        return BlastRadiusEstimate(
            code_change_magnitude=code_magnitude,
            function_importance=func_importance,
            downstream_impact=downstream_impact,
            overall_risk=overall_risk,
            affected_modules=sorted(all_affected),
            risk_level=risk_level,
        )

    def _get_downstream(self, modules: List[str]) -> set:
        """Find all modules transitively depending on the given modules."""
        downstream = set()
        to_check = set(modules)

        while to_check:
            current = to_check.pop()
            for module, deps in self.dependencies.items():
                if current in deps and module not in downstream:
                    downstream.add(module)
                    to_check.add(module)

        # Also add direct dependencies (things that modified modules depend on)
        for mod in modules:
            for dep in self.dependencies.get(mod, []):
                downstream.add(dep)

        return downstream
