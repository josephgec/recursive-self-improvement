"""Condition builder for the Godel Agent ablation suite."""

from __future__ import annotations

from typing import Any, Dict, List

from src.suites.base import AblationCondition


class GodelConditionBuilder:
    """Builds pipeline configurations for Godel Agent conditions."""

    EXPECTED_ACCURACY = {
        "full": 0.85,
        "no_rollback": 0.78,
        "no_validation": 0.76,
        "no_ceiling": 0.80,
        "no_cooldown": 0.81,
        "no_audit": 0.82,
        "no_self_mod": 0.75,
        "unrestricted": 0.70,
    }

    def build(self, condition_name: str) -> Dict[str, Any]:
        """Build pipeline config for a condition."""
        configs = {
            "full": {
                "rollback": True, "validation": True, "ceiling": True,
                "cooldown": True, "audit": True, "self_modification": True,
            },
            "no_rollback": {
                "rollback": False, "validation": True, "ceiling": True,
                "cooldown": True, "audit": True, "self_modification": True,
            },
            "no_validation": {
                "rollback": True, "validation": False, "ceiling": True,
                "cooldown": True, "audit": True, "self_modification": True,
            },
            "no_ceiling": {
                "rollback": True, "validation": True, "ceiling": False,
                "cooldown": True, "audit": True, "self_modification": True,
            },
            "no_cooldown": {
                "rollback": True, "validation": True, "ceiling": True,
                "cooldown": False, "audit": True, "self_modification": True,
            },
            "no_audit": {
                "rollback": True, "validation": True, "ceiling": True,
                "cooldown": True, "audit": False, "self_modification": True,
            },
            "no_self_mod": {
                "rollback": True, "validation": True, "ceiling": True,
                "cooldown": True, "audit": True, "self_modification": False,
            },
            "unrestricted": {
                "rollback": False, "validation": False, "ceiling": False,
                "cooldown": False, "audit": False, "self_modification": True,
            },
        }
        return configs.get(condition_name, configs["full"])

    def get_all_conditions(self) -> List[AblationCondition]:
        conditions = []
        for name, acc in self.EXPECTED_ACCURACY.items():
            conditions.append(AblationCondition(
                name=name,
                description=f"Godel condition: {name} (expected acc={acc})",
                pipeline_config=self.build(name),
                is_full=(name == "full"),
            ))
        return conditions

    def get_expected_accuracy(self, condition_name: str) -> float:
        return self.EXPECTED_ACCURACY.get(condition_name, 0.70)
