"""Markdown audit-report generator.

Assembles a self-contained Markdown report that includes embedded image
references to the temporal-similarity curve and contamination-rate charts,
classifier performance tables, and a final set of recommendations for
downstream consumers of the corpus.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.reporting.curves import LLM_RELEASES, plot_contamination_rate, plot_temporal_similarity_curve

logger = logging.getLogger(__name__)


def _fmt(value: float, decimals: int = 4) -> str:
    """Format a float to a fixed number of decimal places."""
    return f"{value:.{decimals}f}"


def _build_classifier_table(metrics: dict[str, float]) -> str:
    """Return a Markdown table of classifier metrics."""
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
    ]
    display_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 (macro)",
        "auroc": "AUROC",
        "auprc": "AUPRC",
    }
    for key, label in display_names.items():
        if key in metrics:
            lines.append(f"| {label} | {_fmt(metrics[key])} |")
    # Include any extra *numeric* metrics not already shown
    for key, value in metrics.items():
        if key not in display_names and isinstance(value, (int, float)):
            lines.append(f"| {key} | {_fmt(float(value))} |")
    return "\n".join(lines)


def _build_feature_importance_section(metrics: dict) -> str:
    """Return a Markdown section for feature importance if available."""
    importances = metrics.get("feature_importance")
    if importances is None:
        return "_Feature importance data not available._"

    # Accept either a list of dicts or a DataFrame
    if isinstance(importances, pd.DataFrame):
        rows = importances.to_dict("records")
    else:
        rows = list(importances)

    lines = [
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for rank, row in enumerate(rows, start=1):
        name = row.get("feature", "unknown")
        imp = row.get("importance", 0.0)
        lines.append(f"| {rank} | {name} | {_fmt(imp)} |")
    return "\n".join(lines)


def _build_recommendations(alpha_t: float, metrics: dict[str, float]) -> str:
    """Generate recommendations based on audit findings."""
    recs: list[str] = []

    if alpha_t < 0.5:
        recs.append(
            "- **High contamination detected.** Fewer than 50% of documents "
            "pass the authenticity filter. Consider supplementing this corpus "
            "with pre-LLM sources or applying stricter provenance controls."
        )
    elif alpha_t < 0.8:
        recs.append(
            "- **Moderate contamination.** Between 50-80% of documents are "
            "estimated to be human-authored. Temporal sub-setting to pre-2020 "
            "bins may yield a cleaner training set."
        )
    else:
        recs.append(
            "- **Low contamination.** Over 80% of documents appear to be "
            "authentically human-authored. The corpus is suitable for most "
            "downstream uses with standard filtering."
        )

    auroc = metrics.get("auroc", 0.0)
    if auroc < 0.75:
        recs.append(
            "- **Classifier confidence is moderate.** The AUROC is below 0.75, "
            "which means contamination estimates may have significant "
            "uncertainty. Manual spot-checks are recommended."
        )

    recs.append(
        "- Always cross-reference contamination flags with domain-specific "
        "heuristics before discarding documents."
    )
    recs.append(
        "- Re-run this audit periodically as new LLM watermarking and "
        "detection methods become available."
    )

    return "\n".join(recs)


def _build_validation_section(validation_report: dict) -> str:
    """Return a Markdown section for classifier validation results.

    Parameters
    ----------
    validation_report:
        The parsed ``validation_report.json`` dict with keys
        ``test_metrics``, ``per_bin_metrics``, ``feature_importance``,
        ``n_train``, ``n_val``, ``n_test``.

    Returns
    -------
    str
        Markdown text for the validation section.
    """
    parts: list[str] = []

    # Split sizes
    n_train = validation_report.get("n_train", "?")
    n_val = validation_report.get("n_val", "?")
    n_test = validation_report.get("n_test", "?")
    parts.append(
        f"Data split: **{n_train}** train / **{n_val}** validation / "
        f"**{n_test}** held-out test.\n"
    )

    # Test set metrics table
    test_metrics = validation_report.get("test_metrics", {})
    if test_metrics:
        parts.append("### Test Set Metrics\n")
        parts.append(_build_classifier_table(test_metrics))
        parts.append("")

    # Per-bin accuracy table
    per_bin = validation_report.get("per_bin_metrics", [])
    if per_bin:
        parts.append("### Per-Bin Accuracy\n")
        parts.append("| Bin | N docs | Accuracy | Precision | Recall | F1 |")
        parts.append("|-----|--------|----------|-----------|--------|-----|")
        for row in per_bin:
            parts.append(
                f"| {row.get('bin', '')} "
                f"| {row.get('n_docs', '')} "
                f"| {_fmt(row.get('accuracy', 0))} "
                f"| {_fmt(row.get('precision', 0))} "
                f"| {_fmt(row.get('recall', 0))} "
                f"| {_fmt(row.get('f1', 0))} |"
            )
        parts.append("")

    # Calibration curve reference
    parts.append("### Calibration\n")
    parts.append(
        "A reliability diagram showing predicted vs. actual probability "
        "is available below.\n"
    )
    parts.append("![Calibration Curve](../calibration_curve.png)\n")

    # Top 10 feature importance
    feat_imp = validation_report.get("feature_importance", [])
    if feat_imp:
        top10 = feat_imp[:10]
        parts.append("### Top 10 Features\n")
        parts.append("| Rank | Feature | Importance |")
        parts.append("|------|---------|------------|")
        for rank, row in enumerate(top10, start=1):
            name = row.get("feature", "unknown")
            imp = row.get("importance", 0.0)
            parts.append(f"| {rank} | {name} | {_fmt(imp)} |")
        parts.append("")

    return "\n".join(parts)


def generate_audit_report(
    config: dict,
    curve_df: pd.DataFrame,
    classifier_metrics: dict,
    reserve_summary: dict,
    output_dir: Path,
    validation_report: dict | None = None,
    inflection_ci: dict | None = None,
    per_source_curves: bool = False,
) -> Path:
    """Produce a Markdown audit report with embedded chart references.

    Parameters
    ----------
    config:
        Pipeline configuration dictionary.  Used to include the corpus name
        and any relevant settings in the report header.
    curve_df:
        DataFrame from :func:`src.embeddings.temporal_curves.compute_temporal_curve`.
    classifier_metrics:
        Metrics dictionary as returned by
        :meth:`src.classifier.model.ContaminationClassifier.train` (keys:
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``auroc``,
        ``auprc``).  May also include a ``feature_importance`` key holding
        a list of ``{"feature": ..., "importance": ...}`` dicts or a
        DataFrame.
    reserve_summary:
        Summary dictionary as written by
        :func:`src.reserve.export.export_reserve` (keys:
        ``total_documents_audited``, ``reserve_size``, ``alpha_t``,
        ``threshold``, ``mean_authenticity_score``,
        ``temporal_distribution``, ``source_distribution``).
    output_dir:
        Directory where the report Markdown and any generated images will
        be written (created if necessary).
    validation_report:
        Optional dict parsed from ``validation_report.json``.  If provided,
        a "Classifier Validation" section is inserted into the report.
    inflection_ci:
        Optional dict from
        :func:`src.embeddings.temporal_curves.detect_inflection_with_ci`
        with keys ``bin_label``, ``second_derivative``, ``ci_lower``,
        ``ci_upper``, ``confidence``.
    per_source_curves:
        Whether a cross-source comparison plot was generated.

    Returns
    -------
    Path
        Absolute path to the generated ``audit_report.md`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate supporting charts
    # ------------------------------------------------------------------
    curve_img = output_dir / "temporal_similarity_curve.png"
    contamination_img = output_dir / "contamination_rate.png"

    inflection_bin = config.get("inflection_bin")
    llm_releases = config.get("llm_releases")
    plot_temporal_similarity_curve(
        curve_df, curve_img,
        inflection_bin=inflection_bin,
        llm_releases=llm_releases,
    )

    # Compute per-bin synthetic fractions from the reserve summary
    temporal_dist = reserve_summary.get("temporal_distribution", {})
    total_audited = reserve_summary.get("total_documents_audited", 0)
    reserve_size = reserve_summary.get("reserve_size", 0)

    bin_labels = sorted(curve_df["bin"].astype(str).unique())
    synthetic_fractions: list[float] = []
    for b in bin_labels:
        n_reserve = temporal_dist.get(b, 0)
        # n_docs in this bin from the curve
        row = curve_df[curve_df["bin"].astype(str) == b]
        n_total = int(row["n_documents"].iloc[0]) if len(row) > 0 else 0
        if n_total > 0:
            frac_synthetic = 1.0 - (n_reserve / n_total)
            synthetic_fractions.append(max(0.0, frac_synthetic))
        else:
            synthetic_fractions.append(0.0)

    plot_contamination_rate(bin_labels, synthetic_fractions, contamination_img)

    # ------------------------------------------------------------------
    # Extract key values
    # ------------------------------------------------------------------
    corpus_name = config.get("corpus_name", config.get("name", "Unknown Corpus"))
    alpha_t = reserve_summary.get("alpha_t", 0.0)
    threshold = reserve_summary.get("threshold", 0.90)
    mean_auth = reserve_summary.get("mean_authenticity_score", 0.0)
    source_dist = reserve_summary.get("source_distribution", {})
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ------------------------------------------------------------------
    # Assemble Markdown
    # ------------------------------------------------------------------
    sections: list[str] = []

    # Title
    sections.append(f"# Data Contamination Audit Report\n")
    sections.append(f"**Corpus:** {corpus_name}  ")
    sections.append(f"**Generated:** {timestamp}  ")
    sections.append(f"**Authenticity threshold:** {threshold}\n")

    # 1. Executive Summary
    sections.append("## 1. Executive Summary\n")
    sections.append(f"- **Total documents audited:** {total_audited:,}")
    sections.append(f"- **Reserve size (clean documents):** {reserve_size:,}")
    sections.append(
        f"- **Estimated contamination:** {_fmt((1 - alpha_t) * 100, 1)}% "
        f"of the corpus is estimated to be synthetic"
    )
    sections.append(f"- **Alpha_t (clean fraction):** {_fmt(alpha_t)}")
    sections.append(f"- **Mean authenticity score:** {_fmt(mean_auth)}\n")

    # 2. Temporal Similarity Curve
    sections.append("## 2. Temporal Similarity Curve\n")
    sections.append(
        "The following chart shows how intra-bin cosine similarity evolves "
        "across temporal bins. An upward inflection suggests increasing "
        "homogeneity, a hallmark of synthetic contamination.\n"
    )
    sections.append(f"![Temporal Similarity Curve]({curve_img.name})\n")

    if llm_releases:
        release_names = ", ".join(
            f"{r['name']} ({r['date']})" for r in llm_releases
        )
        sections.append(
            f"**LLM Timeline:** The chart is annotated with major LLM release "
            f"dates: {release_names}. Vertical dotted lines mark the "
            f"corresponding time bins for reference.\n"
        )

    # 2a. Inflection Analysis (optional)
    if inflection_ci is not None:
        sections.append("### Inflection Analysis\n")
        sections.append(
            f"- **Inflection point:** {inflection_ci['bin_label']}"
        )
        sections.append(
            f"- **Second derivative:** {_fmt(inflection_ci['second_derivative'], 6)}"
        )
        sections.append(
            f"- **95% CI:** [{inflection_ci['ci_lower']}, "
            f"{inflection_ci['ci_upper']}]"
        )
        sections.append(
            f"- **Bootstrap confidence:** "
            f"{inflection_ci['confidence'] * 100:.1f}% of bootstrap samples "
            f"agree on the modal inflection bin\n"
        )

    # 2b. Cross-Source Comparison (optional)
    if per_source_curves:
        sections.append("### Cross-Source Comparison\n")
        sections.append(
            "The following chart overlays similarity curves for each data "
            "source (e.g., Wikipedia vs. Common Crawl), enabling direct "
            "comparison of contamination trends across sources.\n"
        )
        sections.append("![Cross-Source Comparison](cross_source_comparison.png)\n")

    # 3. Contamination Rate by Year
    sections.append("## 3. Contamination Rate by Year\n")
    sections.append(
        "Per-bin fraction of documents classified as synthetic by the "
        "contamination classifier.\n"
    )
    sections.append(f"![Contamination Rate]({contamination_img.name})\n")

    # 4. Classifier Performance
    sections.append("## 4. Classifier Performance\n")
    sections.append(_build_classifier_table(classifier_metrics))
    sections.append("")

    # Section numbering shifts by 1 when validation report is present
    if validation_report is not None:
        next_num = 5
        sections.append(f"## {next_num}. Classifier Validation\n")
        sections.append(_build_validation_section(validation_report))
        next_num += 1
    else:
        next_num = 5

    # Feature Importance
    sections.append(f"## {next_num}. Feature Importance Ranking\n")
    sections.append(_build_feature_importance_section(classifier_metrics))
    sections.append("")
    next_num += 1

    # Reserve Statistics
    sections.append(f"## {next_num}. Reserve Statistics\n")
    sections.append(f"- **Reserve size:** {reserve_size:,} documents")
    sections.append(f"- **Authenticity threshold:** {threshold}")
    sections.append(f"- **Mean authenticity score:** {_fmt(mean_auth)}")
    if source_dist:
        sections.append("\n**Source distribution:**\n")
        for source, count in sorted(source_dist.items()):
            sections.append(f"- {source}: {count:,}")
    if temporal_dist:
        sections.append("\n**Temporal distribution (reserve documents):**\n")
        for bin_label, count in sorted(temporal_dist.items()):
            sections.append(f"- {bin_label}: {count:,}")
    sections.append("")

    # Recommendations
    next_num += 1
    sections.append(f"## {next_num}. Recommendations\n")
    sections.append(_build_recommendations(alpha_t, classifier_metrics))
    sections.append("")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    report_text = "\n".join(sections)
    report_path = output_dir / "audit_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Generated audit report at %s", report_path)
    return report_path
