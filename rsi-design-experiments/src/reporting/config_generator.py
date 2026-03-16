"""Generates optimal YAML configuration from recommendations."""

import yaml
from typing import Any, Dict, Optional

from src.analysis.recommendation import PipelineRecommendation


class ConfigGenerator:
    """Generates optimal pipeline configuration as YAML."""

    def generate_optimal_config(
        self,
        recommendation: PipelineRecommendation,
    ) -> str:
        """Generate a YAML configuration string from a recommendation.

        Args:
            recommendation: The pipeline recommendation.

        Returns:
            YAML string representing the optimal configuration.
        """
        config = {
            "pipeline": {
                "modification_frequency": {
                    "policy": recommendation.modification_frequency,
                    "confidence": recommendation.confidence_levels.get(
                        "modification_frequency", "unknown"
                    ),
                },
                "hindsight_target": {
                    "policy": recommendation.hindsight_target,
                    "confidence": recommendation.confidence_levels.get(
                        "hindsight_target", "unknown"
                    ),
                },
                "rlm_depth": {
                    "depth": recommendation.rlm_depth,
                    "confidence": recommendation.confidence_levels.get(
                        "rlm_depth", "unknown"
                    ),
                },
            },
            "reasoning": recommendation.reasoning,
            "sensitivity_ranking": recommendation.sensitivity_ranking,
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def generate_minimal_config(
        self,
        recommendation: PipelineRecommendation,
    ) -> str:
        """Generate a minimal YAML configuration with just the settings."""
        config = {
            "modification_frequency": recommendation.modification_frequency,
            "hindsight_target": recommendation.hindsight_target,
            "rlm_depth": recommendation.rlm_depth,
        }
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
