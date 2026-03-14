"""Reporting module: visualisation and audit report generation.

Public API
----------
.. autofunction:: plot_temporal_similarity_curve
.. autofunction:: plot_contamination_rate
.. autofunction:: plot_feature_distributions
.. autofunction:: generate_audit_report
"""

from src.reporting.curves import plot_contamination_rate, plot_temporal_similarity_curve
from src.reporting.distributions import plot_feature_distributions
from src.reporting.summary import generate_audit_report

__all__ = [
    "plot_temporal_similarity_curve",
    "plot_contamination_rate",
    "plot_feature_distributions",
    "generate_audit_report",
]
