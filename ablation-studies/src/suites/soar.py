"""SOAR evolutionary search ablation suite: 8 conditions."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.suites.base import AblationCondition, AblationSuite


class SOARAblation(AblationSuite):
    """Ablation study for the SOAR evolutionary search approach.

    Conditions:
        full: Complete SOAR pipeline
        no_hindsight: Remove hindsight experience replay
        no_crossover: Remove crossover operator
        no_error_guidance: Remove error-guided mutation
        no_mutation: Remove all mutation operators
        random_search: Replace with random search baseline
        single_candidate: Single candidate (no population)
        hindsight_library: Enhanced hindsight with library access
    """

    CONDITION_DEFS = {
        "full": ("Complete SOAR pipeline", True),
        "no_hindsight": ("Without hindsight experience replay", False),
        "no_crossover": ("Without crossover operator", False),
        "no_error_guidance": ("Without error-guided mutation", False),
        "no_mutation": ("Without mutation operators", False),
        "random_search": ("Random search baseline", False),
        "single_candidate": ("Single candidate, no population", False),
        "hindsight_library": ("Enhanced hindsight with library", False),
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
            "hindsight": True,
            "crossover": True,
            "error_guidance": True,
            "mutation": True,
            "population_size": 10,
            "library_access": False,
        }
        overrides = {
            "full": {},
            "no_hindsight": {"hindsight": False},
            "no_crossover": {"crossover": False},
            "no_error_guidance": {"error_guidance": False},
            "no_mutation": {"mutation": False},
            "random_search": {
                "hindsight": False,
                "crossover": False,
                "error_guidance": False,
                "mutation": False,
            },
            "single_candidate": {"population_size": 1, "crossover": False},
            "hindsight_library": {"library_access": True},
        }
        config = {**base, **overrides.get(condition.name, {})}
        return config

    def get_benchmarks(self) -> List[str]:
        return ["search_efficiency", "solution_quality", "diversity"]

    def get_paper_name(self) -> str:
        return "SOAR Evolutionary Search"

    def get_key_comparisons(self) -> List[Tuple[str, str]]:
        return [
            ("full", "random_search"),
            ("full", "no_hindsight"),
            ("full", "no_crossover"),
            ("hindsight_library", "no_hindsight"),
        ]
