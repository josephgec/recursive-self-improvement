"""Statistical analysis for ablation results."""

from src.analysis.statistical_tests import PublicationStatistics, PairwiseResult
from src.analysis.effect_sizes import cohens_d, eta_squared, interpret_d
from src.analysis.confidence_intervals import bootstrap_ci, bootstrap_difference_ci
from src.analysis.interaction_tests import CrossSuiteInteractionAnalyzer, InteractionReport
from src.analysis.power_analysis import required_repetitions, achieved_power, minimum_detectable_effect

__all__ = [
    "PublicationStatistics",
    "PairwiseResult",
    "cohens_d",
    "eta_squared",
    "interpret_d",
    "bootstrap_ci",
    "bootstrap_difference_ci",
    "CrossSuiteInteractionAnalyzer",
    "InteractionReport",
    "required_repetitions",
    "achieved_power",
    "minimum_detectable_effect",
]
