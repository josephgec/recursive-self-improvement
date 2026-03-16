"""Early warning analysis for GDI."""

from typing import Any, Dict, List, Optional, Tuple


class EarlyWarningAnalyzer:
    """Analyzes how far in advance GDI predicts accuracy degradation.

    Computes the lead time between GDI alerts and actual accuracy drops.
    """

    def compute_lead_time(
        self,
        gdi_history: List[float],
        accuracy_history: List[float],
        gdi_threshold: float = 0.40,
        accuracy_threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """Compute lead time between GDI alert and accuracy drop.

        Args:
            gdi_history: List of GDI composite scores.
            accuracy_history: List of accuracy scores (0-1).
            gdi_threshold: GDI score threshold for "alert".
            accuracy_threshold: Accuracy threshold for "degraded".

        Returns:
            Dictionary with lead time analysis.
        """
        if not gdi_history or not accuracy_history:
            return {
                "lead_time_steps": None,
                "gdi_alert_index": None,
                "accuracy_drop_index": None,
                "analysis": "Insufficient data.",
            }

        min_len = min(len(gdi_history), len(accuracy_history))
        gdi_history = gdi_history[:min_len]
        accuracy_history = accuracy_history[:min_len]

        # Find first GDI alert
        gdi_alert_idx = None
        for i, score in enumerate(gdi_history):
            if score >= gdi_threshold:
                gdi_alert_idx = i
                break

        # Find first accuracy drop
        acc_drop_idx = None
        for i, acc in enumerate(accuracy_history):
            if acc < accuracy_threshold:
                acc_drop_idx = i
                break

        if gdi_alert_idx is None or acc_drop_idx is None:
            return {
                "lead_time_steps": None,
                "gdi_alert_index": gdi_alert_idx,
                "accuracy_drop_index": acc_drop_idx,
                "analysis": "Could not compute lead time — "
                           "no alert or no accuracy drop detected.",
            }

        lead_time = acc_drop_idx - gdi_alert_idx

        return {
            "lead_time_steps": lead_time,
            "gdi_alert_index": gdi_alert_idx,
            "accuracy_drop_index": acc_drop_idx,
            "analysis": (
                f"GDI alerted {lead_time} steps before accuracy dropped. "
                f"{'Early warning successful.' if lead_time > 0 else 'No early warning — GDI alert came after or with accuracy drop.'}"
            ),
        }

    def plot_early_warning(
        self,
        gdi_history: List[float],
        accuracy_history: List[float],
    ) -> Dict[str, Any]:
        """Prepare data for early warning visualization.

        Args:
            gdi_history: GDI scores over time.
            accuracy_history: Accuracy scores over time.

        Returns:
            Dictionary with plot data.
        """
        return {
            "gdi_scores": list(gdi_history),
            "accuracy_scores": list(accuracy_history),
            "x_labels": list(range(max(len(gdi_history), len(accuracy_history)))),
        }
