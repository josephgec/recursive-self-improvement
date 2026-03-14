#!/usr/bin/env python
"""Classifier training script.

Loads a pre-computed feature matrix and labels, trains the contamination
classifier, and saves the model to disk.

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
import sys
from pathlib import Path
from typing import Optional

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

    # Configure classifier
    cls_cfg = cfg.get("classifier", {})
    hyperparams = cls_cfg.get("hyperparameters", {})
    val_split = cls_cfg.get("training_data", {}).get("val_split", 0.15)

    console.print(f"[bold]Model type:[/bold] xgboost")
    console.print(f"[bold]Hyperparameters:[/bold] {hyperparams}")
    console.print(f"[bold]Validation split:[/bold] {val_split}")

    # Train
    classifier = ContaminationClassifier(model_type="xgboost", **hyperparams)
    metrics = classifier.train(feature_matrix, labels, val_split=val_split)

    # Save model
    model_path = output_dir / "classifier.joblib"
    classifier.save(model_path)
    console.print(f"[green]Model saved to {model_path}[/green]")

    # Save metrics
    metrics_path = output_dir / "classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"[green]Metrics saved to {metrics_path}[/green]")

    # Display metrics
    console.print("\n[bold]Validation Metrics:[/bold]")
    for key, value in metrics.items():
        console.print(f"  {key}: {value:.4f}")

    # Feature importance
    importance = classifier.feature_importance()
    console.print("\n[bold]Top 10 Features:[/bold]")
    for _, row in importance.head(10).iterrows():
        console.print(f"  {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    app()
