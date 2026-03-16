"""Phase-specific adapters for GDI integration."""

from typing import Any, Dict, List, Optional

from ..composite.gdi import GoalDriftIndex
from ..composite.weights import WeightConfig


class PhaseAdapter:
    """Adapters for different recursive self-improvement phases.

    Each phase may have different sensitivity requirements and
    weight configurations for drift detection.
    """

    @staticmethod
    def for_godel() -> Dict[str, Any]:
        """Configuration for Godel machine phase.

        Emphasizes semantic drift (reasoning changes).
        """
        return {
            "weights": WeightConfig(
                semantic=0.40, lexical=0.20,
                structural=0.15, distributional=0.25
            ),
            "thresholds": {
                "green_max": 0.10,
                "yellow_max": 0.30,
                "orange_max": 0.60,
            },
            "check_interval": 120,
            "phase": "godel",
        }

    @staticmethod
    def for_soar() -> Dict[str, Any]:
        """Configuration for SOAR phase.

        Balanced weights with moderate sensitivity.
        """
        return {
            "weights": WeightConfig(
                semantic=0.30, lexical=0.25,
                structural=0.20, distributional=0.25
            ),
            "thresholds": {
                "green_max": 0.15,
                "yellow_max": 0.40,
                "orange_max": 0.70,
            },
            "check_interval": 300,
            "phase": "soar",
        }

    @staticmethod
    def for_symcode() -> Dict[str, Any]:
        """Configuration for SymCode phase.

        Emphasizes structural drift (code structure changes).
        """
        return {
            "weights": WeightConfig(
                semantic=0.25, lexical=0.20,
                structural=0.35, distributional=0.20
            ),
            "thresholds": {
                "green_max": 0.12,
                "yellow_max": 0.35,
                "orange_max": 0.65,
            },
            "check_interval": 180,
            "phase": "symcode",
        }

    @staticmethod
    def for_rlm() -> Dict[str, Any]:
        """Configuration for RLM (Reinforcement Learning from Memory) phase.

        Emphasizes distributional drift (output distribution changes).
        """
        return {
            "weights": WeightConfig(
                semantic=0.25, lexical=0.20,
                structural=0.15, distributional=0.40
            ),
            "thresholds": {
                "green_max": 0.15,
                "yellow_max": 0.40,
                "orange_max": 0.70,
            },
            "check_interval": 240,
            "phase": "rlm",
        }

    @staticmethod
    def for_pipeline() -> Dict[str, Any]:
        """Configuration for the full pipeline.

        Strict thresholds for end-to-end monitoring.
        """
        return {
            "weights": WeightConfig(
                semantic=0.30, lexical=0.25,
                structural=0.20, distributional=0.25
            ),
            "thresholds": {
                "green_max": 0.10,
                "yellow_max": 0.30,
                "orange_max": 0.55,
            },
            "check_interval": 60,
            "phase": "pipeline",
        }
