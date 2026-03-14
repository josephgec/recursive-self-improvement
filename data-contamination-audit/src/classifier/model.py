"""Contamination classifier: XGBoost-based binary classifier for detecting
AI-generated (synthetic) vs. human-authored documents.

This module trains on the feature matrix produced by
:mod:`src.classifier.features.ensemble` and outputs calibrated probability
estimates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Columns that are metadata, not features — drop before training.
_NON_FEATURE_COLS = {"doc_id", "timestamp"}


class ContaminationClassifier:
    """Binary classifier: 0 = human-authored, 1 = synthetic / AI-generated.

    Parameters
    ----------
    model_type:
        Only ``"xgboost"`` is currently supported.
    **xgb_params:
        Extra keyword arguments forwarded to :class:`xgboost.XGBClassifier`.
    """

    def __init__(self, model_type: str = "xgboost", **xgb_params: Any) -> None:
        if model_type != "xgboost":
            raise ValueError(f"Unsupported model_type: {model_type!r}")
        self.model_type = model_type

        defaults: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        defaults.update(xgb_params)
        self._model = XGBClassifier(**defaults)
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
        """Drop metadata columns that should not be used as features."""
        cols_to_drop = [c for c in df.columns if c in _NON_FEATURE_COLS]
        if cols_to_drop:
            logger.debug("Dropping non-feature columns: %s", cols_to_drop)
        return df.drop(columns=cols_to_drop, errors="ignore")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        val_split: float = 0.15,
    ) -> dict[str, float]:
        """Train the classifier and return evaluation metrics.

        Parameters
        ----------
        features:
            Feature matrix (rows = documents).  ``doc_id`` and ``timestamp``
            columns are automatically dropped.
        labels:
            Binary labels — 0 for human-authored, 1 for synthetic.
        val_split:
            Fraction of data reserved for validation.

        Returns
        -------
        dict[str, float]
            Metrics computed on the held-out validation set:
            ``accuracy``, ``precision``, ``recall``, ``f1``, ``auroc``,
            ``auprc``.
        """
        X = self._drop_non_features(features)
        self._feature_names = list(X.columns)
        y = labels.values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y,
        )

        logger.info(
            "Training %s on %d samples (val=%d)",
            self.model_type,
            len(X_train),
            len(X_val),
        )
        self._model.fit(X_train, y_train)

        # Evaluate on validation set.
        y_pred = self._model.predict(X_val)
        y_prob = self._model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(y_val, y_prob)),
            "auprc": float(average_precision_score(y_val, y_prob)),
        }

        logger.info("Validation metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities.

        Parameters
        ----------
        features:
            Feature matrix with the same columns used during training.

        Returns
        -------
        np.ndarray
            Shape ``(n_docs, 2)`` — columns are ``[p_human, p_synthetic]``.
        """
        X = self._drop_non_features(features)
        return self._model.predict_proba(X)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted by descending importance.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.
        """
        importances = self._model.feature_importances_
        df = pd.DataFrame(
            {"feature": self._feature_names, "importance": importances}
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist classifier to disk.

        Saves a dictionary containing the trained XGBoost model, model type,
        and feature names via :func:`joblib.dump`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "model_type": self.model_type,
            "feature_names": self._feature_names,
        }
        joblib.dump(payload, path)
        logger.info("Saved ContaminationClassifier to %s", path)

    def load(self, path: Path) -> None:
        """Load a previously saved classifier from *path*."""
        path = Path(path)
        payload = joblib.load(path)
        self._model = payload["model"]
        self.model_type = payload["model_type"]
        self._feature_names = payload["feature_names"]
        logger.info("Loaded ContaminationClassifier from %s", path)
