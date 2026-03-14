"""Classifier package — public API re-exports."""

from .calibration import CalibratedClassifier, calibrate, plot_calibration_curve
from .model import ContaminationClassifier

__all__ = [
    "ContaminationClassifier",
    "CalibratedClassifier",
    "calibrate",
    "plot_calibration_curve",
]
