#!/usr/bin/env python
"""End-to-end data contamination audit pipeline.

Usage
-----
    python scripts/run_audit.py --config configs/default.yaml --steps all
    python scripts/run_audit.py --steps download,embed,features
    python scripts/run_audit.py --dry-run
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
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.classifier.features.ensemble import build_feature_matrix
from src.classifier.features.perplexity import PerplexityScorer
from src.classifier.features.watermark import WatermarkDetector
from src.classifier.model import ContaminationClassifier
from src.data.common_crawl import Document
from src.data.sampler import build_temporal_corpus
from src.data.timestamper import assign_time_bin
from src.embeddings.encoder import DocumentEncoder
from src.embeddings.temporal_curves import compute_temporal_curve, detect_inflection_point
from src.reporting.summary import generate_audit_report
from src.reserve.export import export_reserve
from src.reserve.filter import compute_alpha_t, filter_to_reserve
from src.reserve.quality import apply_quality_filters

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
# Pipeline steps
# ---------------------------------------------------------------------------


def step_download(cfg: dict, output_dir: Path) -> None:
    """Step 1: Download Wikipedia and Common Crawl data."""
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

    # Save flattened docs for subsequent steps
    docs_path = output_dir / "all_documents.pkl"
    _save_documents(all_docs, docs_path)

    console.print(
        f"[green]Computed embeddings for {len(all_docs)} documents "
        f"(dim={embeddings.shape[1]}).[/green]"
    )


def step_features(cfg: dict, output_dir: Path) -> None:
    """Step 3: Extract perplexity, watermark, and stylometric features."""
    console.rule("[bold blue]Step 3: Extract Features")

    docs_path = output_dir / "all_documents.pkl"
    all_docs = _load_documents(docs_path)

    ppl_cfg = cfg.get("perplexity", {})
    ppl_model = ppl_cfg.get("model_name", "gpt2")
    stride = ppl_cfg.get("stride", 512)

    perplexity_scorer = PerplexityScorer(model_name=ppl_model, stride=stride)
    watermark_detector = WatermarkDetector()
    tokenizer = perplexity_scorer.tokenizer

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
    """Step 4: Train the contamination classifier."""
    console.rule("[bold blue]Step 4: Train Classifier")

    features_path = output_dir / "feature_matrix.parquet"
    feature_matrix = pd.read_parquet(features_path)

    # Load labels -- in a real pipeline the labels come from the config.
    # For now, look for a labels file. If not found, generate heuristic labels.
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

    cls_cfg = cfg.get("classifier", {})
    hyperparams = cls_cfg.get("hyperparameters", {})
    val_split = cls_cfg.get("training_data", {}).get("val_split", 0.15)

    classifier = ContaminationClassifier(model_type="xgboost", **hyperparams)
    metrics = classifier.train(feature_matrix, labels, val_split=val_split)

    model_path = output_dir / "classifier.joblib"
    classifier.save(model_path)

    metrics_path = output_dir / "classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved classifier metrics to %s", metrics_path)

    console.print(f"[green]Classifier trained. Accuracy: {metrics['accuracy']:.4f}[/green]")


def step_classify(cfg: dict, output_dir: Path) -> None:
    """Step 5: Score all documents with the trained classifier."""
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

    # Apply classifier threshold filter
    reserve_docs = filter_to_reserve(
        documents=all_docs,
        classifier=classifier,
        feature_matrix=feature_matrix,
        threshold=threshold,
    )

    # Apply quality filters
    embeddings_path = output_dir / "embeddings.npy"
    embeddings = np.load(embeddings_path)

    # Build embeddings for reserve docs only
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

    # Compute alpha_t
    alpha_t = compute_alpha_t(len(all_docs), len(reserve_docs))

    # Save reserve
    reserve_dir = output_dir / "reserve"
    _save_documents(reserve_docs, reserve_dir / "reserve_documents.pkl")

    # Export
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

    # Detect inflection point if enough bins
    if len(curve_df) >= 3:
        try:
            inflection_bin = detect_inflection_point(curve_df)
            cfg["inflection_bin"] = inflection_bin
        except ValueError:
            pass

    report_dir = output_dir / "report"
    report_path = generate_audit_report(
        config=cfg,
        curve_df=curve_df,
        classifier_metrics=classifier_metrics,
        reserve_summary=reserve_summary,
        output_dir=report_dir,
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
