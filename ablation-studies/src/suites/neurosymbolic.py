"""Neurosymbolic ablation suite: 7 conditions."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.suites.base import AblationCondition, AblationSuite


class NeurosymbolicAblation(AblationSuite):
    """Ablation study for the neurosymbolic recursive self-improvement approach.

    Conditions:
        full: Complete neurosymbolic pipeline
        symcode_only: Only symbolic code generation (no BDM, no prose)
        bdm_only: Only Bayesian decision model
        prose_only: Only prose-based reasoning (no code, no BDM)
        code_no_verify: Code generation without verification
        hybrid_no_bdm: Hybrid code+prose without BDM selection
        integrative: Full integration with enhanced cross-modal transfer
    """

    CONDITION_DEFS = {
        "full": ("Full neurosymbolic pipeline", True),
        "symcode_only": ("Symbolic code generation only", False),
        "bdm_only": ("Bayesian decision model only", False),
        "prose_only": ("Prose-based reasoning only", False),
        "code_no_verify": ("Code generation without verification", False),
        "hybrid_no_bdm": ("Hybrid code+prose without BDM", False),
        "integrative": ("Full integration with cross-modal transfer", False),
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
            "symbolic_code": True,
            "bdm": True,
            "prose": True,
            "verification": True,
            "cross_modal": False,
        }
        overrides = {
            "full": {},
            "symcode_only": {"bdm": False, "prose": False},
            "bdm_only": {"symbolic_code": False, "prose": False, "verification": False},
            "prose_only": {"symbolic_code": False, "bdm": False, "verification": False},
            "code_no_verify": {"verification": False},
            "hybrid_no_bdm": {"bdm": False},
            "integrative": {"cross_modal": True},
        }
        config = {**base, **overrides.get(condition.name, {})}
        return config

    def get_benchmarks(self) -> List[str]:
        return ["code_generation", "reasoning", "verification"]

    def get_paper_name(self) -> str:
        return "Neurosymbolic Recursive Self-Improvement"

    def get_key_comparisons(self) -> List[Tuple[str, str]]:
        return [
            ("full", "prose_only"),
            ("full", "symcode_only"),
            ("full", "bdm_only"),
            ("symcode_only", "prose_only"),
        ]
