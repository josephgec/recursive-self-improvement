"""Main evaluator — runs all 5 criteria against evidence."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from src.criteria.base import CriterionResult, Evidence, SuccessCriterion
from src.criteria.sustained_improvement import SustainedImprovementCriterion
from src.criteria.paradigm_improvement import ParadigmImprovementCriterion
from src.criteria.gdi_bounds import GDIBoundsCriterion
from src.criteria.publication_acceptance import PublicationAcceptanceCriterion
from src.criteria.auditability import AuditabilityCriterion


class CriteriaEvaluator:
    """Evaluates all success criteria against provided evidence."""

    def __init__(
        self,
        criteria: List[SuccessCriterion] | None = None,
        config: Dict | None = None,
    ):
        if criteria is not None:
            self._criteria = criteria
        else:
            self._criteria = self._build_default_criteria(config or {})

    def _build_default_criteria(
        self, config: Dict
    ) -> List[SuccessCriterion]:
        """Build the default set of 5 criteria with optional config."""
        sustained_cfg = config.get("sustained_improvement", {})
        paradigm_cfg = config.get("paradigm_improvement", {})
        gdi_cfg = config.get("gdi_bounds", {})
        pub_cfg = config.get("publication_acceptance", {})
        audit_cfg = config.get("auditability", {})

        return [
            SustainedImprovementCriterion(
                trend_alpha=sustained_cfg.get("trend_alpha", 0.05),
                min_total_gain_pp=sustained_cfg.get("min_total_gain_pp", 5.0),
                min_collapse_divergence_pp=sustained_cfg.get(
                    "min_collapse_divergence_pp", 10.0
                ),
            ),
            ParadigmImprovementCriterion(
                alpha=paradigm_cfg.get("alpha", 0.05),
                min_effects=paradigm_cfg.get("min_effects"),
            ),
            GDIBoundsCriterion(
                max_gdi=gdi_cfg.get("max_gdi", 0.50),
                max_consecutive_yellow=gdi_cfg.get(
                    "max_consecutive_yellow", 5
                ),
                require_all_phases=gdi_cfg.get(
                    "require_all_phases_monitored", True
                ),
            ),
            PublicationAcceptanceCriterion(
                min_accepted=pub_cfg.get("min_accepted", 2),
                min_tier_1_or_2=pub_cfg.get("min_tier_1_or_2", 1),
                under_review_discount=pub_cfg.get(
                    "under_review_confidence_discount", 0.15
                ),
            ),
            AuditabilityCriterion(
                min_reasoning_traces=audit_cfg.get(
                    "min_reasoning_traces", 20
                ),
                require_hash_chain=audit_cfg.get(
                    "require_hash_chain", True
                ),
            ),
        ]

    def evaluate_all(self, evidence: Evidence) -> List[CriterionResult]:
        """Evaluate all criteria and return results."""
        results = []
        for criterion in self._criteria:
            result = criterion.evaluate(evidence)
            results.append(result)
        return results

    def evaluate_single(
        self, criterion_name: str, evidence: Evidence
    ) -> Optional[CriterionResult]:
        """Evaluate a single criterion by name."""
        for criterion in self._criteria:
            if criterion.name == criterion_name:
                return criterion.evaluate(evidence)
        return None

    @property
    def criteria(self) -> List[SuccessCriterion]:
        """Return the list of criteria."""
        return list(self._criteria)

    @property
    def criteria_names(self) -> List[str]:
        """Return names of all criteria."""
        return [c.name for c in self._criteria]
