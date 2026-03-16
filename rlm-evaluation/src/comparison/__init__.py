"""Comparison tools: cost models, head-to-head, scaling experiments."""

from src.comparison.cost_model import CostModel, CostBreakdown, CostComparison
from src.comparison.head_to_head import HeadToHeadComparator, HeadToHeadReport
from src.comparison.scaling_experiment import ContextScalingExperiment, ScalingResult
from src.comparison.efficiency_frontier import EfficiencyFrontierAnalyzer
from src.comparison.statistical_tests import StatisticalTests

__all__ = [
    "CostModel",
    "CostBreakdown",
    "CostComparison",
    "HeadToHeadComparator",
    "HeadToHeadReport",
    "ContextScalingExperiment",
    "ScalingResult",
    "EfficiencyFrontierAnalyzer",
    "StatisticalTests",
]
