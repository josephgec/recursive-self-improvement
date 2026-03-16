"""Godel Agent ablation suite: 8 conditions."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.suites.base import AblationCondition, AblationSuite


class GodelAgentAblation(AblationSuite):
    """Ablation study for the Godel Agent self-modification approach.

    Conditions:
        full: Complete Godel agent with all safety mechanisms
        no_rollback: Remove rollback capability
        no_validation: Remove pre-modification validation
        no_ceiling: Remove performance ceiling bounds
        no_cooldown: Remove cooldown between modifications
        no_audit: Remove audit trail logging
        no_self_mod: Disable self-modification entirely
        unrestricted: Remove all safety constraints (expected to degrade)
    """

    CONDITION_DEFS = {
        "full": ("Complete Godel agent with all safety mechanisms", True),
        "no_rollback": ("Without rollback capability", False),
        "no_validation": ("Without pre-modification validation", False),
        "no_ceiling": ("Without performance ceiling bounds", False),
        "no_cooldown": ("Without cooldown between modifications", False),
        "no_audit": ("Without audit trail logging", False),
        "no_self_mod": ("Self-modification disabled entirely", False),
        "unrestricted": ("All safety constraints removed", False),
    }

    def get_conditions(self) -> List[AblationCondition]:
        conditions = []
        for name, (desc, is_full) in self.CONDITION_DEFS.items():
            conditions.append(AblationCondition(
                name=name,
                description=desc,
                pipeline_config=self.build_pipeline(
                    AblationCondition(name=name, description=desc, is_full=is_full)
                ),
                is_full=is_full,
            ))
        return conditions

    def build_pipeline(self, condition: AblationCondition) -> Dict[str, Any]:
        base = {
            "rollback": True,
            "validation": True,
            "ceiling": True,
            "cooldown": True,
            "audit": True,
            "self_modification": True,
        }
        overrides = {
            "full": {},
            "no_rollback": {"rollback": False},
            "no_validation": {"validation": False},
            "no_ceiling": {"ceiling": False},
            "no_cooldown": {"cooldown": False},
            "no_audit": {"audit": False},
            "no_self_mod": {"self_modification": False},
            "unrestricted": {
                "rollback": False,
                "validation": False,
                "ceiling": False,
                "cooldown": False,
                "audit": False,
            },
        }
        config = {**base, **overrides.get(condition.name, {})}
        return config

    def get_benchmarks(self) -> List[str]:
        return ["self_modification", "safety_bounds", "convergence"]

    def get_paper_name(self) -> str:
        return "Godel Agent Self-Modification"

    def get_key_comparisons(self) -> List[Tuple[str, str]]:
        return [
            ("full", "no_self_mod"),
            ("full", "unrestricted"),
            ("full", "no_rollback"),
            ("full", "no_validation"),
        ]
