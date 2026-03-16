"""RLM recursive language model ablation suite: 8 conditions."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.suites.base import AblationCondition, AblationSuite


class RLMAblation(AblationSuite):
    """Ablation study for recursive language model scaffolding.

    Conditions:
        full: Complete RLM pipeline
        no_recursion: Disable recursive decomposition
        no_helpers: Disable helper function generation
        depth_1: Limit recursion depth to 1
        no_repl: Disable REPL-based execution
        repl_no_code: REPL without code execution
        chunked_prompt: Chunked prompting baseline
        rag_baseline: RAG-based retrieval baseline
    """

    CONDITION_DEFS = {
        "full": ("Complete RLM pipeline", True),
        "no_recursion": ("Without recursive decomposition", False),
        "no_helpers": ("Without helper function generation", False),
        "depth_1": ("Recursion depth limited to 1", False),
        "no_repl": ("Without REPL-based execution", False),
        "repl_no_code": ("REPL without code execution", False),
        "chunked_prompt": ("Chunked prompting baseline", False),
        "rag_baseline": ("RAG retrieval baseline", False),
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
            "recursion": True,
            "helpers": True,
            "max_depth": 5,
            "repl": True,
            "code_execution": True,
            "chunking": False,
            "rag": False,
        }
        overrides = {
            "full": {},
            "no_recursion": {"recursion": False, "max_depth": 0},
            "no_helpers": {"helpers": False},
            "depth_1": {"max_depth": 1},
            "no_repl": {"repl": False, "code_execution": False},
            "repl_no_code": {"code_execution": False},
            "chunked_prompt": {
                "recursion": False,
                "max_depth": 0,
                "chunking": True,
            },
            "rag_baseline": {
                "recursion": False,
                "max_depth": 0,
                "rag": True,
            },
        }
        config = {**base, **overrides.get(condition.name, {})}
        return config

    def get_benchmarks(self) -> List[str]:
        return ["large_context", "multi_step", "code_execution"]

    def get_paper_name(self) -> str:
        return "Recursive Language Model Scaffolding"

    def get_key_comparisons(self) -> List[Tuple[str, str]]:
        return [
            ("full", "no_repl"),
            ("full", "no_recursion"),
            ("full", "depth_1"),
            ("full", "rag_baseline"),
        ]
