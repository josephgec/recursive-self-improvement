"""Success criteria definitions."""

from src.criteria.base import SuccessCriterion, CriterionResult, Evidence
from src.criteria.sustained_improvement import SustainedImprovementCriterion
from src.criteria.paradigm_improvement import ParadigmImprovementCriterion
from src.criteria.gdi_bounds import GDIBoundsCriterion
from src.criteria.publication_acceptance import PublicationAcceptanceCriterion
from src.criteria.auditability import AuditabilityCriterion

ALL_CRITERIA = [
    SustainedImprovementCriterion,
    ParadigmImprovementCriterion,
    GDIBoundsCriterion,
    PublicationAcceptanceCriterion,
    AuditabilityCriterion,
]

__all__ = [
    "SuccessCriterion",
    "CriterionResult",
    "Evidence",
    "SustainedImprovementCriterion",
    "ParadigmImprovementCriterion",
    "GDIBoundsCriterion",
    "PublicationAcceptanceCriterion",
    "AuditabilityCriterion",
    "ALL_CRITERIA",
]
