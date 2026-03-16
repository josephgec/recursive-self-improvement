"""Dashboard configuration for GDI monitoring."""

from typing import Any, Dict, List


class DashboardConfig:
    """Configuration for monitoring dashboards.

    Provides panel definitions that can be used with W&B or other
    visualization tools.
    """

    @staticmethod
    def get_panels() -> List[Dict[str, Any]]:
        """Get panel definitions for the GDI dashboard.

        Returns:
            List of panel configuration dictionaries.
        """
        return [
            {
                "title": "GDI Composite Score",
                "type": "line",
                "metric": "gdi/composite_score",
                "y_range": [0.0, 1.0],
                "thresholds": {
                    "green": 0.15,
                    "yellow": 0.40,
                    "orange": 0.70,
                },
            },
            {
                "title": "Signal Decomposition",
                "type": "stacked_area",
                "metrics": [
                    "gdi/semantic_score",
                    "gdi/lexical_score",
                    "gdi/structural_score",
                    "gdi/distributional_score",
                ],
                "y_range": [0.0, 1.0],
            },
            {
                "title": "Alert Level Timeline",
                "type": "discrete",
                "metric": "gdi/alert_level",
                "colors": {
                    "green": "#00cc00",
                    "yellow": "#ffcc00",
                    "orange": "#ff6600",
                    "red": "#ff0000",
                },
            },
            {
                "title": "Trend Indicator",
                "type": "text",
                "metric": "gdi/trend",
            },
            {
                "title": "Signal Components — Semantic",
                "type": "multi_line",
                "metrics": [
                    "gdi/semantic/centroid_distance",
                    "gdi/semantic/pairwise_drift",
                    "gdi/semantic/mmd",
                ],
            },
            {
                "title": "Signal Components — Lexical",
                "type": "multi_line",
                "metrics": [
                    "gdi/lexical/js_divergence",
                    "gdi/lexical/vocabulary_shift",
                    "gdi/lexical/ngram_novelty",
                ],
            },
            {
                "title": "Signal Components — Structural",
                "type": "multi_line",
                "metrics": [
                    "gdi/structural/sentence_length_shift",
                    "gdi/structural/depth_distribution_shift",
                    "gdi/structural/node_type_shift",
                ],
            },
            {
                "title": "Signal Components — Distributional",
                "type": "multi_line",
                "metrics": [
                    "gdi/distributional/kl_forward",
                    "gdi/distributional/kl_reverse",
                    "gdi/distributional/total_variation",
                    "gdi/distributional/js_divergence",
                ],
            },
        ]
