"""Probability calibration for the contamination classifier.

Provides Platt scaling via :class:`sklearn.calibration.CalibratedClassifierCV`
and a helper to generate reliability diagrams.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator

from .model import ContaminationClassifier

logger = logging.getLogger(__name__)

# Use non-interactive backend so plots can be saved without a display.
matplotlib.use("Agg")


class CalibratedClassifier:
    """Wrapper that combines a :class:`ContaminationClassifier` with a
    calibration model so that :meth:`predict_proba` returns calibrated
    probabilities.

    This object is produced by :func:`calibrate` and should generally not
    be instantiated directly.
    """

    def __init__(
        self,
        base_classifier: ContaminationClassifier,
        calibrator: CalibratedClassifierCV,
        feature_names: list[str],
    ) -> None:
        self.base_classifier = base_classifier
        self._calibrator = calibrator
        self._feature_names = feature_names

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities.

        Parameters
        ----------
        features:
            Feature matrix (same format as training data).

        Returns
        -------
        np.ndarray
            Shape ``(n_docs, 2)`` — columns are ``[p_human, p_synthetic]``.
        """
        X = ContaminationClassifier._drop_non_features(features)
        return self._calibrator.predict_proba(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the calibrated classifier to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_classifier": self.base_classifier,
            "calibrator": self._calibrator,
            "feature_names": self._feature_names,
        }
        joblib.dump(payload, path)
        logger.info("Saved CalibratedClassifier to %s", path)

    @classmethod
    def load(cls, path: Path) -> "CalibratedClassifier":
        """Load a previously saved calibrated classifier."""
        path = Path(path)
        payload = joblib.load(path)
        logger.info("Loaded CalibratedClassifier from %s", path)
        return cls(
            base_classifier=payload["base_classifier"],
            calibrator=payload["calibrator"],
            feature_names=payload["feature_names"],
        )


# ======================================================================
# Public functions
# ======================================================================


def calibrate(
    classifier: ContaminationClassifier,
    val_features: pd.DataFrame,
    val_labels: pd.Series,
) -> CalibratedClassifier:
    """Apply Platt scaling to an already-trained classifier.

    Wraps the trained XGBoost model in a :class:`sklearn.frozen.FrozenEstimator`
    so that :class:`sklearn.calibration.CalibratedClassifierCV` does not
    re-fit it — only the sigmoid calibration layer is fit on the supplied
    validation data.

    Parameters
    ----------
    classifier:
        A trained :class:`ContaminationClassifier`.
    val_features:
        Validation feature matrix (should not overlap with training data).
    val_labels:
        Binary labels for the validation set.

    Returns
    -------
    CalibratedClassifier
        A wrapper whose :meth:`predict_proba` returns calibrated
        probabilities.
    """
    X_val = ContaminationClassifier._drop_non_features(val_features)
    feature_names = list(X_val.columns)
    y_val = val_labels.values

    logger.info(
        "Calibrating classifier with Platt scaling on %d validation samples",
        len(y_val),
    )

    # Wrap in FrozenEstimator so CalibratedClassifierCV will not re-train
    # the base model (sklearn >= 1.6 replacement for cv="prefit").
    frozen = FrozenEstimator(classifier._model)
    cal = CalibratedClassifierCV(
        estimator=frozen,
        method="sigmoid",
        cv=2,
    )
    cal.fit(X_val, y_val)

    return CalibratedClassifier(
        base_classifier=classifier,
        calibrator=cal,
        feature_names=feature_names,
    )


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    *,
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Diagram)",
) -> None:
    """Generate and save a reliability diagram.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_prob:
        Predicted probabilities for the positive class (synthetic).
    output_path:
        File path to save the plot (e.g. ``"calibration.png"``).
    n_bins:
        Number of bins for the calibration curve.
    title:
        Plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform",
    )

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]},
    )

    # -- Top: reliability diagram -----------------------------------------
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="Classifier",
        color="#1f77b4",
    )
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # -- Bottom: histogram of predicted probabilities ---------------------
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved calibration curve to %s", output_path)
