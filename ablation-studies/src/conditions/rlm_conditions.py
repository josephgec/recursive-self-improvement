"""Condition builder for the RLM ablation suite."""

from __future__ import annotations

from typing import Any, Dict, List

from src.suites.base import AblationCondition


class RLMConditionBuilder:
    """Builds pipeline configurations for RLM conditions."""

    EXPECTED_ACCURACY = {
        "full": 0.85,
        "no_recursion": 0.73,
        "no_helpers": 0.78,
        "depth_1": 0.76,
        "no_repl": 0.60,
        "repl_no_code": 0.68,
        "chunked_prompt": 0.70,
        "rag_baseline": 0.72,
    }

    def build(self, condition_name: str) -> Dict[str, Any]:
        """Build pipeline config for a condition."""
        configs = {
            "full": {
                "recursion": True, "helpers": True, "max_depth": 5,
                "repl": True, "code_execution": True, "chunking": False, "rag": False,
            },
            "no_recursion": {
                "recursion": False, "helpers": True, "max_depth": 0,
                "repl": True, "code_execution": True, "chunking": False, "rag": False,
            },
            "no_helpers": {
                "recursion": True, "helpers": False, "max_depth": 5,
                "repl": True, "code_execution": True, "chunking": False, "rag": False,
            },
            "depth_1": {
                "recursion": True, "helpers": True, "max_depth": 1,
                "repl": True, "code_execution": True, "chunking": False, "rag": False,
            },
            "no_repl": {
                "recursion": True, "helpers": True, "max_depth": 5,
                "repl": False, "code_execution": False, "chunking": False, "rag": False,
            },
            "repl_no_code": {
                "recursion": True, "helpers": True, "max_depth": 5,
                "repl": True, "code_execution": False, "chunking": False, "rag": False,
            },
            "chunked_prompt": {
                "recursion": False, "helpers": True, "max_depth": 0,
                "repl": True, "code_execution": True, "chunking": True, "rag": False,
            },
            "rag_baseline": {
                "recursion": False, "helpers": True, "max_depth": 0,
                "repl": True, "code_execution": True, "chunking": False, "rag": True,
            },
        }
        return configs.get(condition_name, configs["full"])

    def get_all_conditions(self) -> List[AblationCondition]:
        conditions = []
        for name, acc in self.EXPECTED_ACCURACY.items():
            conditions.append(AblationCondition(
                name=name,
                description=f"RLM condition: {name} (expected acc={acc})",
                pipeline_config=self.build(name),
                is_full=(name == "full"),
            ))
        return conditions

    def get_expected_accuracy(self, condition_name: str) -> float:
        return self.EXPECTED_ACCURACY.get(condition_name, 0.60)
