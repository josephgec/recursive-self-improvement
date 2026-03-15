#!/usr/bin/env python
"""Classifier training script with validation.

Loads a pre-computed feature matrix and labels, splits into train/val/test,
trains the contamination classifier, runs calibration, computes per-bin
accuracy, and saves a full validation report.

Usage
-----
    python scripts/train_classifier.py \\
        --features-path data/feature_matrix.parquet \\
        --labels-path data/labels.csv \\
        --output-dir data/
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.classifier.model import ContaminationClassifier

console = Console()
logger = logging.getLogger("train_classifier")

app = typer.Typer(help="Train the contamination classifier.")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-bin evaluation
# ---------------------------------------------------------------------------


def per_bin_evaluation(
    classifier: ContaminationClassifier,
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    doc_ids: pd.Series | np.ndarray,
    docs_path: Path,
) -> pd.DataFrame:
    """Compute accuracy, precision, recall, f1 per time_bin.

    Parameters
    ----------
    classifier:
        Trained :class:`ContaminationClassifier`.
    features:
        Feature matrix for the evaluation set.
    labels:
        True binary labels for the evaluation set.
    doc_ids:
        Document IDs corresponding to the evaluation set rows.
    docs_path:
        Path to the pickled documents list (must contain ``time_bin``
        in each document's ``metadata``).

    Returns
    -------
    pd.DataFrame
        Columns: ``bin``, ``n_docs``, ``accuracy``, ``precision``,
        ``recall``, ``f1``.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # Load docs to get time_bin metadata
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)

    # Build doc_id -> time_bin mapping
    id_to_bin: dict[str, str] = {}
    for doc in docs:
        tb = doc.metadata.get("time_bin")
        if tb is not None:
            id_to_bin[doc.doc_id] = str(tb)

    # Predict
    y_pred = classifier._model.predict(
        ContaminationClassifier._drop_non_features(features)
    )
    y_true = np.asarray(labels)
    ids = np.asarray(doc_ids)

    # Group by time_bin
    rows: list[dict] = []
    bins_for_ids = [id_to_bin.get(str(did), "__unknown__") for did in ids]
    eval_df = pd.DataFrame({
        "bin": bins_for_ids,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    for bin_label, grp in eval_df.groupby("bin"):
        if bin_label == "__unknown__":
            continue
        yt = grp["y_true"].values
        yp = grp["y_pred"].values
        rows.append({
            "bin": bin_label,
            "n_docs": len(grp),
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values("bin").reset_index(drop=True)
    return result


@app.command()
def main(
    config: Path = typer.Option(
        _PROJECT_ROOT / "configs" / "default.yaml",
        "--config",
        help="Path to the pipeline configuration YAML file.",
    ),
    features_path: Path = typer.Option(
        ...,
        "--features-path",
        help="Path to the feature matrix Parquet file.",
    ),
    labels_path: Path = typer.Option(
        ...,
        "--labels-path",
        help="Path to the labels CSV file (must contain a 'label' column).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Directory to save the trained model. Defaults to data/.",
    ),
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        help="Path to pickled documents for per-bin evaluation.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Train the contamination classifier on pre-computed features."""
    _setup_logging(verbose)

    cfg = _load_config(config)
    logger.info("Loaded config from %s", config)

    # Resolve output directory
    if output_dir is None:
        output_dir = Path(cfg.get("project", {}).get("output_dir", "data"))
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load feature matrix
    console.print(f"[bold]Loading features from:[/bold] {features_path}")
    feature_matrix = pd.read_parquet(features_path)
    console.print(f"  Shape: {feature_matrix.shape}")

    # Load labels
    console.print(f"[bold]Loading labels from:[/bold] {labels_path}")
    labels_df = pd.read_csv(labels_path)
    if "label" not in labels_df.columns:
        console.print("[red]Error: labels CSV must contain a 'label' column.[/red]")
        raise typer.Exit(code=1)
    labels = labels_df["label"]

    if len(labels) != len(feature_matrix):
        console.print(
            f"[red]Error: feature matrix has {len(feature_matrix)} rows "
            f"but labels have {len(labels)} entries.[/red]"
        )
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # a) Train/test split: 70% train+val, 15% val (internal), 15% test
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split

    # First split: 85% train+val, 15% held-out test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.15,
        random_state=42,
        stratify=labels,
    )

    features_train_val = feature_matrix.iloc[train_val_idx].reset_index(drop=True)
    labels_train_val = labels.iloc[train_val_idx].reset_index(drop=True)
    features_test = feature_matrix.iloc[test_idx].reset_index(drop=True)
    labels_test = labels.iloc[test_idx].reset_index(drop=True)

    # ContaminationClassifier.train() does its own internal train/val split.
    # We pass val_split so it carves ~15% of the 85% (= ~12.75% of total)
    # as its internal validation set, leaving ~72.25% for actual training.
    cls_cfg = cfg.get("classifier", {})
    hyperparams = cls_cfg.get("hyperparameters", {})
    # Internal val_split: 15/85 ~ 0.176 to get ~15% of total as val
    internal_val_split = round(15.0 / 85.0, 3)

    console.print(f"[bold]Model type:[/bold] xgboost")
    console.print(f"[bold]Hyperparameters:[/bold] {hyperparams}")
    console.print(
        f"[bold]Split:[/bold] {len(train_val_idx)} train+val, "
        f"{len(test_idx)} held-out test "
        f"(internal val_split={internal_val_split})"
    )

    # Train
    classifier = ContaminationClassifier(model_type="xgboost", **hyperparams)
    val_metrics = classifier.train(
        features_train_val, labels_train_val, val_split=internal_val_split,
    )

    # Save model
    model_path = output_dir / "classifier.joblib"
    classifier.save(model_path)
    console.print(f"[green]Model saved to {model_path}[/green]")

    # ------------------------------------------------------------------
    # Evaluate on held-out test set
    # ------------------------------------------------------------------
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    X_test = ContaminationClassifier._drop_non_features(features_test)
    y_test = labels_test.values
    y_test_pred = classifier._model.predict(X_test)
    y_test_prob = classifier._model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_test, y_test_prob)),
        "auprc": float(average_precision_score(y_test, y_test_prob)),
    }

    console.print("\n[bold]Test Set Metrics:[/bold]")
    for key, value in test_metrics.items():
        console.print(f"  {key}: {value:.4f}")

    # ------------------------------------------------------------------
    # b) Per-bin accuracy analysis
    # ------------------------------------------------------------------
    per_bin_metrics_list: list[dict] = []

    # Resolve docs_path: explicit flag, or default location
    if docs_path is None:
        candidate = output_dir / "all_documents.pkl"
        if candidate.exists():
            docs_path = candidate

    if docs_path is not None and docs_path.exists():
        console.print(f"\n[bold]Per-bin evaluation using:[/bold] {docs_path}")
        test_doc_ids = feature_matrix.iloc[test_idx]["doc_id"] if "doc_id" in feature_matrix.columns else None
        if test_doc_ids is not None:
            per_bin_df = per_bin_evaluation(
                classifier, features_test, labels_test,
                test_doc_ids.reset_index(drop=True), docs_path,
            )
            if len(per_bin_df) > 0:
                per_bin_metrics_list = per_bin_df.to_dict("records")
                console.print("\n[bold]Per-Bin Test Accuracy:[/bold]")
                for row in per_bin_metrics_list:
                    console.print(
                        f"  {row['bin']}: n={row['n_docs']}, "
                        f"acc={row['accuracy']:.4f}, f1={row['f1']:.4f}"
                    )
        else:
            logger.info("No doc_id column in features — skipping per-bin analysis.")
    else:
        logger.info("No documents pickle found — skipping per-bin analysis.")

    # ------------------------------------------------------------------
    # c) Feature importance
    # ------------------------------------------------------------------
    importance = classifier.feature_importance()
    importance_path = output_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    console.print(f"\n[green]Feature importance saved to {importance_path}[/green]")

    console.print("\n[bold]Top 10 Features:[/bold]")
    for _, row in importance.head(10).iterrows():
        console.print(f"  {row['feature']}: {row['importance']:.4f}")

    # ------------------------------------------------------------------
    # d) Calibration
    # ------------------------------------------------------------------
    # Lazy import to avoid xgboost/torch conflict at module level
    from src.classifier.calibration import calibrate, plot_calibration_curve

    # Re-split train_val the same way the classifier did internally to
    # recover the validation indices for calibration fitting.
    # train_test_split returns (train, test) — we want the *test* portion
    # as the calibration set.
    _, cal_idx = train_test_split(
        np.arange(len(labels_train_val)),
        test_size=internal_val_split,
        random_state=42,
        stratify=labels_train_val,
    )
    features_cal = features_train_val.iloc[cal_idx].reset_index(drop=True)
    labels_cal = labels_train_val.iloc[cal_idx].reset_index(drop=True)

    calibrated = calibrate(classifier, features_cal, labels_cal)
    calibrated_path = output_dir / "classifier_calibrated.joblib"
    calibrated.save(calibrated_path)
    console.print(f"[green]Calibrated model saved to {calibrated_path}[/green]")

    # Evaluate calibration on test set
    cal_probs = calibrated.predict_proba(features_test)[:, 1]
    cal_curve_path = output_dir / "calibration_curve.png"
    plot_calibration_curve(y_test, cal_probs, cal_curve_path)
    console.print(f"[green]Calibration curve saved to {cal_curve_path}[/green]")

    # ------------------------------------------------------------------
    # e) Save validation report as JSON
    # ------------------------------------------------------------------
    feature_importance_records = importance.to_dict("records")

    # Compute split sizes.  The internal val split carved ~15/85 of
    # train_val, so approximate counts:
    n_internal_val = len(cal_idx)
    n_internal_train = len(labels_train_val) - n_internal_val

    validation_report = {
        "test_metrics": test_metrics,
        "per_bin_metrics": per_bin_metrics_list,
        "feature_importance": feature_importance_records,
        "n_train": int(n_internal_train),
        "n_val": int(n_internal_val),
        "n_test": int(len(labels_test)),
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    console.print(f"[green]Validation report saved to {report_path}[/green]")

    # Also save the older-style classifier_metrics.json for backward compat
    metrics_path = output_dir / "classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    console.print(f"[green]Classifier metrics saved to {metrics_path}[/green]")


if __name__ == "__main__":
    app()
