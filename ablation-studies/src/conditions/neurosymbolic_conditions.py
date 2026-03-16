"""Condition builder for the neurosymbolic ablation suite."""

from __future__ import annotations

from typing import Any, Dict, List

from src.suites.base import AblationCondition


class NeurosymbolicConditionBuilder:
    """Builds pipeline configurations for neurosymbolic conditions."""

    # Expected accuracy for each condition (used by MockPipeline)
    EXPECTED_ACCURACY = {
        "full": 0.85,
        "symcode_only": 0.78,
        "bdm_only": 0.74,
        "prose_only": 0.72,
        "code_no_verify": 0.79,
        "hybrid_no_bdm": 0.80,
        "integrative": 0.83,
    }

    def build(self, condition_name: str) -> Dict[str, Any]:
        """Build pipeline config for a condition."""
        configs = {
            "full": {
                "symbolic_code": True,
                "bdm": True,
                "prose": True,
                "verification": True,
                "cross_modal": False,
            },
            "symcode_only": {
                "symbolic_code": True,
                "bdm": False,
                "prose": False,
                "verification": True,
                "cross_modal": False,
            },
            "bdm_only": {
                "symbolic_code": False,
                "bdm": True,
                "prose": False,
                "verification": False,
                "cross_modal": False,
            },
            "prose_only": {
                "symbolic_code": False,
                "bdm": False,
                "prose": True,
                "verification": False,
                "cross_modal": False,
            },
            "code_no_verify": {
                "symbolic_code": True,
                "bdm": True,
                "prose": True,
                "verification": False,
                "cross_modal": False,
            },
            "hybrid_no_bdm": {
                "symbolic_code": True,
                "bdm": False,
                "prose": True,
                "verification": True,
                "cross_modal": False,
            },
            "integrative": {
                "symbolic_code": True,
                "bdm": True,
                "prose": True,
                "verification": True,
                "cross_modal": True,
            },
        }
        return configs.get(condition_name, configs["full"])

    def get_all_conditions(self) -> List[AblationCondition]:
        """Build all conditions for the suite."""
        conditions = []
        for name, acc in self.EXPECTED_ACCURACY.items():
            conditions.append(AblationCondition(
                name=name,
                description=f"Neurosymbolic condition: {name} (expected acc={acc})",
                pipeline_config=self.build(name),
                is_full=(name == "full"),
            ))
        return conditions

    def get_expected_accuracy(self, condition_name: str) -> float:
        """Return expected accuracy for a condition."""
        return self.EXPECTED_ACCURACY.get(condition_name, 0.70)
