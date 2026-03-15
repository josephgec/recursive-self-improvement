#!/usr/bin/env python
"""End-to-end data contamination audit pipeline.

Usage
-----
    python scripts/run_audit.py --config configs/default.yaml --steps all
    python scripts/run_audit.py --steps download,embed,features
    python scripts/run_audit.py --dry-run

NOTE: Imports are deferred into each step function to avoid a known conflict
between xgboost and torch GPT-2 inference on macOS ARM64.  The ``features``
step (which runs GPT-2 perplexity scoring) must execute before xgboost is
loaded into the process.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.common_crawl import Document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

console = Console()
logger = logging.getLogger("run_audit")

ALL_STEPS = ["download", "embed", "features", "train", "classify", "filter", "report"]

app = typer.Typer(help="Run the data contamination audit pipeline end-to-end.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _load_config(config_path: Path) -> dict:
    """Load a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _save_documents(docs: list[Document], path: Path) -> None:
    """Pickle documents to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(docs, f)
    logger.info("Saved %d documents to %s", len(docs), path)


def _load_documents(path: Path) -> list[Document]:
    """Load pickled documents from disk."""
    with open(path, "rb") as f:
        docs = pickle.load(f)
    logger.info("Loaded %d documents from %s", len(docs), path)
    return docs


def _save_corpus(corpus: dict[str, list[Document]], path: Path) -> None:
    """Pickle a temporal corpus to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(corpus, f)
    logger.info("Saved corpus (%d bins) to %s", len(corpus), path)


def _load_corpus(path: Path) -> dict[str, list[Document]]:
    """Load a pickled temporal corpus from disk."""
    with open(path, "rb") as f:
        corpus = pickle.load(f)
    logger.info("Loaded corpus (%d bins) from %s", len(corpus), path)
    return corpus


def _flatten_corpus(corpus: dict[str, list[Document]]) -> list[Document]:
    """Flatten a temporal corpus into a single document list."""
    all_docs: list[Document] = []
    for bin_label, docs in sorted(corpus.items()):
        for doc in docs:
            doc.metadata["time_bin"] = bin_label
        all_docs.extend(docs)
    return all_docs


# ---------------------------------------------------------------------------
# Pipeline steps — imports are deferred to avoid xgboost/torch conflicts
# ---------------------------------------------------------------------------


def step_download(cfg: dict, output_dir: Path) -> None:
    """Step 1: Download Wikipedia and Common Crawl data."""
    from src.data.sampler import build_temporal_corpus

    console.rule("[bold blue]Step 1: Download Data")

    sampling_cfg = cfg.get("sampling", {})
    sources = sampling_cfg.get("sources", ["wikipedia", "common_crawl"])
    n_per_bin = sampling_cfg.get("n_per_bin", 5000)
    bins = sampling_cfg.get("bins", [])
    bin_size = sampling_cfg.get("bin_size", "year")
    seed = cfg.get("project", {}).get("seed", 42)

    corpus = build_temporal_corpus(
        sources=sources,
        n_per_bin=n_per_bin,
        bins=bins,
        seed=seed,
        bin_size=bin_size,
    )

    corpus_path = output_dir / "corpus.pkl"
    _save_corpus(corpus, corpus_path)

    total = sum(len(v) for v in corpus.values())
    console.print(
        f"[green]Downloaded {total} documents across {len(corpus)} time bins.[/green]"
    )


def step_embed(cfg: dict, output_dir: Path) -> None:
    """Step 2: Compute embeddings for all documents."""
    from src.embeddings.encoder import DocumentEncoder
    from src.embeddings.temporal_curves import compute_per_source_curves, compute_temporal_curve

    console.rule("[bold blue]Step 2: Compute Embeddings")

    corpus_path = output_dir / "corpus.pkl"
    corpus = _load_corpus(corpus_path)
    all_docs = _flatten_corpus(corpus)

    emb_cfg = cfg.get("embeddings", {})
    model_name = emb_cfg.get("model_name", "all-MiniLM-L6-v2")
    batch_size = emb_cfg.get("batch_size", 256)
    cache_dir = emb_cfg.get("cache_dir")
    if cache_dir:
        cache_dir = output_dir / cache_dir
    else:
        cache_dir = output_dir / "embeddings"

    encoder = DocumentEncoder(
        model_name=model_name,
        batch_size=batch_size,
    )

    embeddings = encoder.encode(all_docs, cache_dir=cache_dir)

    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info("Saved embeddings (%s) to %s", embeddings.shape, embeddings_path)

    # Also compute the temporal curve
    sim_cfg = cfg.get("similarity", {})
    reference_bin = sim_cfg.get("reference_bin")
    curve_df = compute_temporal_curve(corpus, encoder, reference_bin=reference_bin)

    curve_path = output_dir / "temporal_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    logger.info("Saved temporal curve to %s", curve_path)

    # Compute per-source temporal curves
    per_source = compute_per_source_curves(corpus, encoder, reference_bin=reference_bin)
    for source_name, source_curve in per_source.items():
        source_curve_path = output_dir / f"temporal_curve_{source_name}.csv"
        source_curve.to_csv(source_curve_path, index=False)
        logger.info("Saved per-source curve for %s to %s", source_name, source_curve_path)

    # Save a manifest of available per-source curves
    per_source_manifest = list(per_source.keys())
    manifest_path = output_dir / "per_source_curves.json"
    with open(manifest_path, "w") as f:
        json.dump(per_source_manifest, f)
    logger.info("Saved per-source curves manifest to %s", manifest_path)

    # Save flattened docs for subsequent steps
    docs_path = output_dir / "all_documents.pkl"
    _save_documents(all_docs, docs_path)

    console.print(
        f"[green]Computed embeddings for {len(all_docs)} documents "
        f"(dim={embeddings.shape[1]}).[/green]"
    )


def step_features(cfg: dict, output_dir: Path) -> None:
    """Step 3: Extract perplexity, watermark, and stylometric features.

    IMPORTANT: This step must run BEFORE step_train/step_classify/step_filter
    because those steps import xgboost, which conflicts with torch GPT-2
    inference on macOS ARM64.
    """
    from src.classifier.features.ensemble import build_feature_matrix
    from src.classifier.features.perplexity import PerplexityScorer
    from src.classifier.features.watermark import WatermarkDetector

    console.rule("[bold blue]Step 3: Extract Features")

    docs_path = output_dir / "all_documents.pkl"
    all_docs = _load_documents(docs_path)

    ppl_cfg = cfg.get("perplexity", {})
    ppl_model = ppl_cfg.get("model_name", "gpt2")
    stride = ppl_cfg.get("stride", 512)

    if cfg.get("skip_perplexity", False):
        logger.info("Skipping perplexity scoring (--skip-perplexity flag set)")
        perplexity_scorer = None
        # Still need a tokenizer for the watermark detector.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ppl_model)
    else:
        perplexity_scorer = PerplexityScorer(model_name=ppl_model, stride=stride)
        tokenizer = perplexity_scorer.tokenizer

    watermark_detector = WatermarkDetector()

    features_cache = output_dir / "features_cache"
    feature_matrix = build_feature_matrix(
        documents=all_docs,
        perplexity_scorer=perplexity_scorer,
        watermark_detector=watermark_detector,
        tokenizer=tokenizer,
        cache_dir=features_cache,
    )

    features_path = output_dir / "feature_matrix.parquet"
    feature_matrix.to_parquet(features_path, index=False)
    logger.info("Saved feature matrix (%s) to %s", feature_matrix.shape, features_path)

    console.print(
        f"[green]Extracted {feature_matrix.shape[1] - 1} features "
        f"for {len(all_docs)} documents.[/green]"
    )


def step_train(cfg: dict, output_dir: Path) -> None:
    """Step 4: Train the contamination classifier with validation."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    from src.classifier.model import ContaminationClassifier

    console.rule("[bold blue]Step 4: Train Classifier")

    # ------------------------------------------------------------------
    # Resolve features and labels — prefer real training data if available
    # ------------------------------------------------------------------
    training_dir = output_dir.parent / "data" / "training"
    # Also check relative to project root
    project_training_dir = _PROJECT_ROOT / "data" / "training"

    real_labels_path = None
    real_features_path = None

    for candidate_dir in [training_dir, project_training_dir]:
        candidate_labels = candidate_dir / "labels.csv"
        candidate_features = candidate_dir / "features.parquet"
        if candidate_labels.exists() and candidate_features.exists():
            real_labels_path = candidate_labels
            real_features_path = candidate_features
            break

    if real_features_path is not None and real_labels_path is not None:
        console.print(
            f"[bold cyan]Using real training data from:[/bold cyan] "
            f"{real_features_path.parent}"
        )
        feature_matrix = pd.read_parquet(real_features_path)
        labels_df = pd.read_csv(real_labels_path)
        labels = labels_df["label"]
    else:
        features_path = output_dir / "feature_matrix.parquet"
        feature_matrix = pd.read_parquet(features_path)

        labels_path = output_dir / "labels.csv"
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
            labels = labels_df["label"]
        else:
            logger.warning(
                "No labels file found at %s. Using heuristic labels based on "
                "stylometric features (vocabulary_richness < median -> synthetic).",
                labels_path,
            )
            median_richness = feature_matrix["vocabulary_richness"].median()
            labels = (feature_matrix["vocabulary_richness"] < median_richness).astype(int)

    # ------------------------------------------------------------------
    # Train/test split: 85% train+val, 15% held-out test
    # ------------------------------------------------------------------
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

    cls_cfg = cfg.get("classifier", {})
    hyperparams = cls_cfg.get("hyperparameters", {})
    # Internal val_split: 15/85 ~ 0.176 to get ~15% of total as val
    internal_val_split = round(15.0 / 85.0, 3)

    classifier = ContaminationClassifier(model_type="xgboost", **hyperparams)
    val_metrics = classifier.train(
        features_train_val, labels_train_val, val_split=internal_val_split,
    )

    model_path = output_dir / "classifier.joblib"
    classifier.save(model_path)

    # ------------------------------------------------------------------
    # Evaluate on held-out test set
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Per-bin evaluation
    # ------------------------------------------------------------------
    per_bin_metrics_list: list[dict] = []
    docs_path = output_dir / "all_documents.pkl"
    if docs_path.exists() and "doc_id" in feature_matrix.columns:
        from scripts.train_classifier import per_bin_evaluation

        test_doc_ids = feature_matrix.iloc[test_idx]["doc_id"].reset_index(drop=True)
        per_bin_df = per_bin_evaluation(
            classifier, features_test, labels_test,
            test_doc_ids, docs_path,
        )
        if len(per_bin_df) > 0:
            per_bin_metrics_list = per_bin_df.to_dict("records")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    importance = classifier.feature_importance()
    importance_path = output_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    from src.classifier.calibration import calibrate, plot_calibration_curve

    # Recover the internal validation portion for calibration fitting
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

    cal_probs = calibrated.predict_proba(features_test)[:, 1]
    cal_curve_path = output_dir / "calibration_curve.png"
    plot_calibration_curve(y_test, cal_probs, cal_curve_path)

    # ------------------------------------------------------------------
    # Save validation report
    # ------------------------------------------------------------------
    n_internal_val = len(cal_idx)
    n_internal_train = len(labels_train_val) - n_internal_val

    validation_report = {
        "test_metrics": test_metrics,
        "per_bin_metrics": per_bin_metrics_list,
        "feature_importance": importance.to_dict("records"),
        "n_train": int(n_internal_train),
        "n_val": int(n_internal_val),
        "n_test": int(len(labels_test)),
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    logger.info("Saved validation report to %s", report_path)

    # Also save backward-compatible classifier_metrics.json
    metrics_path = output_dir / "classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    logger.info("Saved classifier metrics to %s", metrics_path)

    console.print(
        f"[green]Classifier trained. Test accuracy: "
        f"{test_metrics['accuracy']:.4f}[/green]"
    )


def step_classify(cfg: dict, output_dir: Path) -> None:
    """Step 5: Score all documents with the trained classifier."""
    from src.classifier.model import ContaminationClassifier

    console.rule("[bold blue]Step 5: Classify Documents")

    features_path = output_dir / "feature_matrix.parquet"
    feature_matrix = pd.read_parquet(features_path)

    model_path = output_dir / "classifier.joblib"
    classifier = ContaminationClassifier()
    classifier.load(model_path)

    probas = classifier.predict_proba(feature_matrix)

    scores_df = pd.DataFrame({
        "doc_id": feature_matrix["doc_id"],
        "p_human": probas[:, 0],
        "p_synthetic": probas[:, 1],
    })

    scores_path = output_dir / "classification_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info("Saved classification scores to %s", scores_path)

    mean_p_synthetic = float(probas[:, 1].mean())
    console.print(
        f"[green]Classified {len(feature_matrix)} documents. "
        f"Mean P(synthetic): {mean_p_synthetic:.4f}[/green]"
    )


def step_filter(cfg: dict, output_dir: Path) -> None:
    """Step 6: Apply threshold + quality filters to build reserve."""
    from src.classifier.model import ContaminationClassifier
    from src.reserve.export import export_reserve
    from src.reserve.filter import compute_alpha_t, filter_to_reserve
    from src.reserve.quality import apply_quality_filters

    console.rule("[bold blue]Step 6: Filter to Reserve")

    docs_path = output_dir / "all_documents.pkl"
    all_docs = _load_documents(docs_path)

    features_path = output_dir / "feature_matrix.parquet"
    feature_matrix = pd.read_parquet(features_path)

    model_path = output_dir / "classifier.joblib"
    classifier = ContaminationClassifier()
    classifier.load(model_path)

    reserve_cfg = cfg.get("reserve", {})
    threshold = reserve_cfg.get("threshold", 0.90)

    reserve_docs = filter_to_reserve(
        documents=all_docs,
        classifier=classifier,
        feature_matrix=feature_matrix,
        threshold=threshold,
    )

    # Apply quality filters
    embeddings_path = output_dir / "embeddings.npy"
    embeddings = np.load(embeddings_path)

    reserve_ids = {doc.doc_id for doc in reserve_docs}
    reserve_indices = [
        i for i, doc in enumerate(all_docs) if doc.doc_id in reserve_ids
    ]
    reserve_embeddings = embeddings[reserve_indices]

    quality_config = {
        "dedup_threshold": reserve_cfg.get("dedup_threshold", 0.95),
        "target_lang": "en",
        "min_chars": reserve_cfg.get("min_document_length", 500),
        "max_chars": reserve_cfg.get("max_document_length", 500_000),
    }
    reserve_docs = apply_quality_filters(
        reserve_docs, reserve_embeddings, quality_config,
    )

    alpha_t = compute_alpha_t(len(all_docs), len(reserve_docs))

    reserve_dir = output_dir / "reserve"
    _save_documents(reserve_docs, reserve_dir / "reserve_documents.pkl")

    export_reserve(
        documents=reserve_docs,
        output_dir=reserve_dir,
        format="parquet",
        full_corpus_size=len(all_docs),
        threshold=threshold,
    )

    console.print(
        f"[green]Reserve: {len(reserve_docs)} / {len(all_docs)} documents "
        f"(alpha_t = {alpha_t:.4f}).[/green]"
    )


def step_report(cfg: dict, output_dir: Path) -> None:
    """Step 7: Generate temporal curves and summary report."""
    from src.embeddings.temporal_curves import detect_inflection_point, detect_inflection_with_ci
    from src.reporting.curves import LLM_RELEASES, plot_cross_source_comparison
    from src.reporting.summary import generate_audit_report

    console.rule("[bold blue]Step 7: Generate Report")

    curve_path = output_dir / "temporal_curve.csv"
    curve_df = pd.read_csv(curve_path)

    metrics_path = output_dir / "classifier_metrics.json"
    with open(metrics_path) as f:
        classifier_metrics = json.load(f)

    reserve_dir = output_dir / "reserve"
    summary_path = reserve_dir / "summary.json"
    with open(summary_path) as f:
        reserve_summary = json.load(f)

    if len(curve_df) >= 3:
        try:
            inflection_bin = detect_inflection_point(curve_df)
            cfg["inflection_bin"] = inflection_bin
        except ValueError:
            pass

    # Compute inflection point with bootstrap confidence interval
    inflection_ci = None
    if len(curve_df) >= 3:
        try:
            inflection_ci = detect_inflection_with_ci(curve_df)
        except ValueError:
            logger.warning("Could not compute inflection CI (too few bins)")

    # Load per-source curves if available
    per_source_curves: dict[str, pd.DataFrame] = {}
    manifest_path = output_dir / "per_source_curves.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            source_names = json.load(f)
        for source_name in source_names:
            source_curve_path = output_dir / f"temporal_curve_{source_name}.csv"
            if source_curve_path.exists():
                per_source_curves[source_name] = pd.read_csv(source_curve_path)

    # Generate cross-source comparison plot if per-source data exists
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    has_per_source = len(per_source_curves) > 1
    if has_per_source:
        cross_source_img = report_dir / "cross_source_comparison.png"
        plot_cross_source_comparison(
            per_source_curves, cross_source_img, llm_releases=LLM_RELEASES,
        )

    # Pass LLM_RELEASES to config so generate_audit_report can use them
    cfg["llm_releases"] = LLM_RELEASES

    # Load validation report if it exists
    validation_report = None
    validation_report_path = output_dir / "validation_report.json"
    if validation_report_path.exists():
        with open(validation_report_path) as f:
            validation_report = json.load(f)
        logger.info("Loaded validation report from %s", validation_report_path)

    report_path = generate_audit_report(
        config=cfg,
        curve_df=curve_df,
        classifier_metrics=classifier_metrics,
        reserve_summary=reserve_summary,
        output_dir=report_dir,
        validation_report=validation_report,
        inflection_ci=inflection_ci,
        per_source_curves=has_per_source,
    )

    console.print(f"[green]Audit report generated at {report_path}[/green]")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

STEP_FUNCTIONS = {
    "download": step_download,
    "embed": step_embed,
    "features": step_features,
    "train": step_train,
    "classify": step_classify,
    "filter": step_filter,
    "report": step_report,
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    config: Path = typer.Option(
        _PROJECT_ROOT / "configs" / "default.yaml",
        "--config",
        help="Path to the pipeline configuration YAML file.",
    ),
    steps: str = typer.Option(
        "all",
        "--steps",
        help=(
            "Comma-separated list of pipeline steps to run, or 'all'. "
            "Options: download, embed, features, train, classify, filter, report"
        ),
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory (overrides config). Defaults to data/.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would be done without executing.",
    ),
    skip_perplexity: bool = typer.Option(
        False,
        "--skip-perplexity",
        help="Skip GPT-2 perplexity scoring for faster runs.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run the data contamination audit pipeline end-to-end."""
    _setup_logging(verbose)

    # Load configuration
    cfg = _load_config(config)
    cfg["skip_perplexity"] = skip_perplexity
    logger.info("Loaded config from %s", config)

    # Resolve output directory
    if output_dir is None:
        output_dir = Path(cfg.get("project", {}).get("output_dir", "data"))
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse steps
    if steps.strip().lower() == "all":
        selected_steps = ALL_STEPS
    else:
        selected_steps = [s.strip().lower() for s in steps.split(",")]
        unknown = [s for s in selected_steps if s not in STEP_FUNCTIONS]
        if unknown:
            console.print(f"[red]Unknown steps: {unknown}[/red]")
            console.print(f"[red]Valid steps: {ALL_STEPS}[/red]")
            raise typer.Exit(code=1)

    console.print(f"[bold]Pipeline steps:[/bold] {', '.join(selected_steps)}")
    console.print(f"[bold]Output directory:[/bold] {output_dir}")
    console.print(f"[bold]Config:[/bold] {config}")

    if dry_run:
        console.print("\n[yellow][DRY RUN] The following steps would execute:[/yellow]")
        for step_name in selected_steps:
            console.print(f"  - {step_name}")
        console.print("[yellow]No actions taken.[/yellow]")
        return

    # Execute steps
    for step_name in selected_steps:
        try:
            STEP_FUNCTIONS[step_name](cfg, output_dir)
        except Exception:
            console.print(f"[red]Step '{step_name}' failed![/red]")
            logger.exception("Step '%s' failed with exception:", step_name)
            raise typer.Exit(code=1)

    console.print("\n[bold green]Pipeline complete.[/bold green]")


if __name__ == "__main__":
    app()
