"""Ablation conditions: define and configure ablation study conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AblationCondition:
    """A single ablation condition."""
    name: str
    description: str
    enabled_components: List[str]
    disabled_components: List[str]
    expected_behavior: str  # "improving", "degrading", "slow_improving"
    metadata: Dict[str, Any] = field(default_factory=dict)


ALL_COMPONENTS = ["soar", "ctm", "godel", "rlm"]


def build_all_conditions() -> List[AblationCondition]:
    """Build the 7 standard ablation conditions."""
    return [
        AblationCondition(
            name="full_pipeline",
            description="Full RSI pipeline with all components",
            enabled_components=list(ALL_COMPONENTS),
            disabled_components=[],
            expected_behavior="improving",
        ),
        AblationCondition(
            name="no_soar",
            description="Pipeline without SOAR meta-learning",
            enabled_components=["ctm", "godel", "rlm"],
            disabled_components=["soar"],
            expected_behavior="slow_improving",
        ),
        AblationCondition(
            name="no_ctm",
            description="Pipeline without CTM conscious processing",
            enabled_components=["soar", "godel", "rlm"],
            disabled_components=["ctm"],
            expected_behavior="slow_improving",
        ),
        AblationCondition(
            name="no_godel",
            description="Pipeline without Godel self-reference",
            enabled_components=["soar", "ctm", "rlm"],
            disabled_components=["godel"],
            expected_behavior="slow_improving",
        ),
        AblationCondition(
            name="no_rlm",
            description="Pipeline without RLM reinforcement",
            enabled_components=["soar", "ctm", "godel"],
            disabled_components=["rlm"],
            expected_behavior="slow_improving",
        ),
        AblationCondition(
            name="soar_only",
            description="Only SOAR meta-learning enabled",
            enabled_components=["soar"],
            disabled_components=["ctm", "godel", "rlm"],
            expected_behavior="slow_improving",
        ),
        AblationCondition(
            name="naive_self_train",
            description="Naive self-training without RSI components",
            enabled_components=[],
            disabled_components=list(ALL_COMPONENTS),
            expected_behavior="degrading",
        ),
    ]


def configure_pipeline_for_condition(
    condition: AblationCondition,
) -> Dict[str, Any]:
    """Return pipeline configuration for a given ablation condition."""
    config: Dict[str, Any] = {
        "condition_name": condition.name,
        "enabled": {comp: True for comp in condition.enabled_components},
        "disabled": {comp: True for comp in condition.disabled_components},
    }
    for comp in ALL_COMPONENTS:
        config[f"use_{comp}"] = comp in condition.enabled_components
    return config
