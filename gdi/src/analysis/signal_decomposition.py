"""Signal decomposition analysis for GDI."""

from typing import Any, Dict, List, Optional

from ..composite.gdi import GDIResult


class SignalDecompositionAnalyzer:
    """Analyzes the contribution of each signal to the GDI score.

    Helps identify which aspect of drift is driving the composite score.
    """

    def decompose(self, gdi_result: GDIResult) -> Dict[str, Any]:
        """Decompose a GDI result into signal contributions.

        Args:
            gdi_result: GDI computation result.

        Returns:
            Dictionary with per-signal contributions and analysis.
        """
        total = gdi_result.composite_score
        if total == 0:
            return {
                "contributions": {
                    "semantic": 0.0,
                    "lexical": 0.0,
                    "structural": 0.0,
                    "distributional": 0.0,
                },
                "primary_driver": "none",
                "analysis": "No drift detected.",
            }

        scores = {
            "semantic": gdi_result.semantic_score,
            "lexical": gdi_result.lexical_score,
            "structural": gdi_result.structural_score,
            "distributional": gdi_result.distributional_score,
        }

        # Relative contribution of each signal
        contributions = {k: v / max(total, 1e-10) for k, v in scores.items()}

        primary = max(scores, key=scores.get)

        return {
            "contributions": contributions,
            "scores": scores,
            "primary_driver": primary,
            "analysis": f"Primary drift driver: {primary} "
                        f"(score={scores[primary]:.3f})",
        }

    def identify_primary_driver(
        self, history: List[GDIResult]
    ) -> str:
        """Identify the primary drift driver across history.

        Args:
            history: List of GDI results.

        Returns:
            Signal name that most frequently drives drift.
        """
        if not history:
            return "none"

        driver_counts: Dict[str, int] = {
            "semantic": 0, "lexical": 0,
            "structural": 0, "distributional": 0,
        }

        for result in history:
            decomp = self.decompose(result)
            driver = decomp["primary_driver"]
            if driver in driver_counts:
                driver_counts[driver] += 1

        return max(driver_counts, key=driver_counts.get)

    def plot_signal_trajectories(
        self, history: List[GDIResult]
    ) -> Dict[str, List[float]]:
        """Extract signal trajectories for plotting.

        Args:
            history: List of GDI results.

        Returns:
            Dictionary mapping signal names to score lists.
        """
        trajectories: Dict[str, List[float]] = {
            "composite": [],
            "semantic": [],
            "lexical": [],
            "structural": [],
            "distributional": [],
        }

        for result in history:
            trajectories["composite"].append(result.composite_score)
            trajectories["semantic"].append(result.semantic_score)
            trajectories["lexical"].append(result.lexical_score)
            trajectories["structural"].append(result.structural_score)
            trajectories["distributional"].append(result.distributional_score)

        return trajectories
