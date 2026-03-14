"""Classifier package.

Imports are lazy to avoid loading xgboost at module import time,
which conflicts with torch GPT-2 inference on macOS ARM64.
"""

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ContaminationClassifier":
        from .model import ContaminationClassifier
        return ContaminationClassifier
    if name == "CalibratedClassifier":
        from .calibration import CalibratedClassifier
        return CalibratedClassifier
    if name == "calibrate":
        from .calibration import calibrate
        return calibrate
    if name == "plot_calibration_curve":
        from .calibration import plot_calibration_curve
        return plot_calibration_curve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContaminationClassifier",
    "CalibratedClassifier",
    "calibrate",
    "plot_calibration_curve",
]
