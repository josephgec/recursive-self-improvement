"""Condition builder for the SOAR ablation suite."""

from __future__ import annotations

from typing import Any, Dict, List

from src.suites.base import AblationCondition


class SOARConditionBuilder:
    """Builds pipeline configurations for SOAR conditions."""

    EXPECTED_ACCURACY = {
        "full": 0.85,
        "no_hindsight": 0.77,
        "no_crossover": 0.79,
        "no_error_guidance": 0.78,
        "no_mutation": 0.73,
        "random_search": 0.65,
        "single_candidate": 0.71,
        "hindsight_library": 0.84,
    }

    def build(self, condition_name: str) -> Dict[str, Any]:
        """Build pipeline config for a condition."""
        configs = {
            "full": {
                "hindsight": True, "crossover": True, "error_guidance": True,
                "mutation": True, "population_size": 10, "library_access": False,
            },
            "no_hindsight": {
                "hindsight": False, "crossover": True, "error_guidance": True,
                "mutation": True, "population_size": 10, "library_access": False,
            },
            "no_crossover": {
                "hindsight": True, "crossover": False, "error_guidance": True,
                "mutation": True, "population_size": 10, "library_access": False,
            },
            "no_error_guidance": {
                "hindsight": True, "crossover": True, "error_guidance": False,
                "mutation": True, "population_size": 10, "library_access": False,
            },
            "no_mutation": {
                "hindsight": True, "crossover": True, "error_guidance": True,
                "mutation": False, "population_size": 10, "library_access": False,
            },
            "random_search": {
                "hindsight": False, "crossover": False, "error_guidance": False,
                "mutation": False, "population_size": 10, "library_access": False,
            },
            "single_candidate": {
                "hindsight": True, "crossover": False, "error_guidance": True,
                "mutation": True, "population_size": 1, "library_access": False,
            },
            "hindsight_library": {
                "hindsight": True, "crossover": True, "error_guidance": True,
                "mutation": True, "population_size": 10, "library_access": True,
            },
        }
        return configs.get(condition_name, configs["full"])

    def get_all_conditions(self) -> List[AblationCondition]:
        conditions = []
        for name, acc in self.EXPECTED_ACCURACY.items():
            conditions.append(AblationCondition(
                name=name,
                description=f"SOAR condition: {name} (expected acc={acc})",
                pipeline_config=self.build(name),
                is_full=(name == "full"),
            ))
        return conditions

    def get_expected_accuracy(self, condition_name: str) -> float:
        return self.EXPECTED_ACCURACY.get(condition_name, 0.65)
