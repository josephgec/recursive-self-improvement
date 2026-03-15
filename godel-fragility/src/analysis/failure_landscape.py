"""Analyze the failure landscape across scenarios and conditions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.harness.stress_runner import ScenarioResult, StressTestResults
from src.measurement.failure_classifier import FailureMode


@dataclass
class Vulnerability:
    """A specific vulnerability identified in the agent."""

    name: str
    category: str
    severity: int  # 1-5
    description: str
    failure_mode: FailureMode
    affected_scenarios: List[str] = field(default_factory=list)
    remediation: str = ""


@dataclass
class FailureLandscape:
    """Full failure landscape analysis."""

    failure_mode_counts: Dict[str, int] = field(default_factory=dict)
    category_failure_rates: Dict[str, float] = field(default_factory=dict)
    severity_distribution: Dict[int, int] = field(default_factory=dict)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    heatmap_data: Optional[Dict[str, Any]] = None
    total_scenarios: int = 0
    total_failures: int = 0


class FailureLandscapeAnalyzer:
    """Analyze failure patterns across stress test results."""

    def compute_landscape(
        self, results: StressTestResults
    ) -> FailureLandscape:
        """Compute the full failure landscape from stress test results."""
        landscape = FailureLandscape()
        landscape.total_scenarios = results.total_scenarios

        # Count failure modes
        mode_counts: Dict[str, int] = defaultdict(int)
        for r in results.results:
            if not r.success:
                mode_counts[r.failure_mode.value] += 1
                landscape.total_failures += 1
        landscape.failure_mode_counts = dict(mode_counts)

        # Category failure rates
        category_total: Dict[str, int] = defaultdict(int)
        category_failed: Dict[str, int] = defaultdict(int)
        for r in results.results:
            category_total[r.category] += 1
            if not r.success:
                category_failed[r.category] += 1

        landscape.category_failure_rates = {
            cat: category_failed[cat] / category_total[cat]
            for cat in category_total
        }

        # Severity distribution
        sev_dist: Dict[int, int] = defaultdict(int)
        for r in results.results:
            if not r.success:
                sev_dist[r.severity] += 1
        landscape.severity_distribution = dict(sev_dist)

        # Build heatmap data: category x failure_mode
        heatmap: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in results.results:
            if not r.success:
                heatmap[r.category][r.failure_mode.value] += 1
        landscape.heatmap_data = {
            cat: dict(modes) for cat, modes in heatmap.items()
        }

        # Identify vulnerabilities
        landscape.vulnerabilities = self.identify_critical_vulnerabilities(results)

        return landscape

    def plot_failure_heatmap(
        self, landscape: FailureLandscape, output_path: Optional[str] = None
    ) -> Optional[Any]:
        """Generate a failure heatmap visualization.

        Returns the matplotlib figure, or None if plotting is not available.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        if not landscape.heatmap_data:
            return None

        categories = sorted(landscape.heatmap_data.keys())
        all_modes = sorted(
            set(
                mode
                for modes in landscape.heatmap_data.values()
                for mode in modes
            )
        )

        if not categories or not all_modes:
            return None

        matrix = np.zeros((len(categories), len(all_modes)))
        for i, cat in enumerate(categories):
            for j, mode in enumerate(all_modes):
                matrix[i, j] = landscape.heatmap_data.get(cat, {}).get(mode, 0)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(all_modes)))
        ax.set_xticklabels(all_modes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=8)

        plt.colorbar(im, ax=ax, label="Count")
        ax.set_title("Failure Mode Heatmap by Category")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150)

        return fig

    def identify_critical_vulnerabilities(
        self, results: StressTestResults
    ) -> List[Vulnerability]:
        """Identify the most critical vulnerabilities from results."""
        vulnerabilities: List[Vulnerability] = []

        # Group failures by failure mode
        mode_scenarios: Dict[str, List[ScenarioResult]] = defaultdict(list)
        for r in results.results:
            if not r.success:
                mode_scenarios[r.failure_mode.value].append(r)

        severity_map = {
            "self_lobotomy": 5,
            "state_corruption": 5,
            "silent_degradation": 4,
            "runaway_modification": 4,
            "rollback_failure": 4,
            "complexity_explosion": 3,
            "infinite_loop": 3,
            "oscillation": 3,
            "stagnation": 2,
            "rollback_partial": 2,
            "validation_caught": 1,
            "deliberation_avoided": 1,
        }

        for mode, scenarios in mode_scenarios.items():
            sev = severity_map.get(mode, 3)
            if sev >= 3:
                vuln = Vulnerability(
                    name=f"{mode}_vulnerability",
                    category=scenarios[0].category if scenarios else "unknown",
                    severity=sev,
                    description=f"Agent exhibits {mode} in {len(scenarios)} scenario(s)",
                    failure_mode=FailureMode(mode),
                    affected_scenarios=[s.scenario_name for s in scenarios],
                    remediation=self._suggest_remediation(mode),
                )
                vulnerabilities.append(vuln)

        # Sort by severity descending
        vulnerabilities.sort(key=lambda v: v.severity, reverse=True)
        return vulnerabilities

    def _suggest_remediation(self, mode: str) -> str:
        """Suggest remediation for a failure mode."""
        suggestions = {
            "self_lobotomy": "Add immutable core functions that cannot be self-modified.",
            "state_corruption": "Implement state integrity checks and sandboxed execution.",
            "silent_degradation": "Add continuous regression testing after each modification.",
            "runaway_modification": "Implement modification rate limiting and cooldown periods.",
            "rollback_failure": "Use append-only checkpoints with integrity verification.",
            "complexity_explosion": "Set hard complexity limits and auto-refactor triggers.",
            "infinite_loop": "Add loop detection and maximum iteration bounds.",
            "oscillation": "Implement change dampening or minimum improvement thresholds.",
            "stagnation": "Add exploration incentives or random restarts.",
        }
        return suggestions.get(mode, "Review agent architecture for this failure pattern.")
