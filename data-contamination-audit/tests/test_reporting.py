"""Tests for the reporting module: curve plots, distribution plots, and audit reports.

All tests use the matplotlib 'Agg' backend for headless (non-interactive)
rendering, and write to a temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Force headless backend before any other matplotlib import.
matplotlib.use("Agg")

from src.reporting import (
    generate_audit_report,
    plot_contamination_rate,
    plot_feature_distributions,
    plot_temporal_similarity_curve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def curve_df() -> pd.DataFrame:
    """Synthetic temporal-curve DataFrame matching the compute_temporal_curve schema."""
    np.random.seed(42)
    bins = [str(y) for y in range(2013, 2026)]
    n = len(bins)
    # Simulate a rising similarity trend with an inflection around 2020
    base = np.linspace(0.30, 0.45, n)
    inflection_bump = np.concatenate([np.zeros(7), np.linspace(0.0, 0.15, n - 7)])
    mean_sim = base + inflection_bump + np.random.normal(0, 0.005, n)
    cross_sim = base + 0.02 + np.random.normal(0, 0.005, n)

    return pd.DataFrame({
        "bin": bins,
        "mean_similarity": mean_sim,
        "cross_similarity_to_reference": cross_sim,
        "n_documents": np.random.randint(80, 200, size=n),
        "similarity_p25": mean_sim - 0.05,
        "similarity_p75": mean_sim + 0.05,
    })


@pytest.fixture
def classifier_metrics() -> dict:
    """Classifier metrics dict matching ContaminationClassifier.train() output."""
    return {
        "accuracy": 0.9120,
        "precision": 0.8850,
        "recall": 0.9300,
        "f1": 0.9070,
        "auroc": 0.9650,
        "auprc": 0.9410,
        "feature_importance": [
            {"feature": "perplexity_mean", "importance": 0.32},
            {"feature": "watermark_z_score", "importance": 0.25},
            {"feature": "vocabulary_richness", "importance": 0.18},
            {"feature": "sentence_length_std", "importance": 0.15},
            {"feature": "repetition_ratio", "importance": 0.10},
        ],
    }


@pytest.fixture
def reserve_summary(curve_df: pd.DataFrame) -> dict:
    """Reserve summary dict matching export.export_reserve() output."""
    total = int(curve_df["n_documents"].sum())
    reserve = int(total * 0.72)
    temporal_dist = {}
    for _, row in curve_df.iterrows():
        # ~70% of each bin survives into the reserve
        temporal_dist[str(row["bin"])] = int(row["n_documents"] * 0.70)
    return {
        "total_documents_audited": total,
        "reserve_size": reserve,
        "alpha_t": reserve / total,
        "threshold": 0.90,
        "mean_authenticity_score": 0.87,
        "temporal_distribution": temporal_dist,
        "source_distribution": {"common_crawl": reserve - 200, "wikipedia": 200},
        "generation_timestamp": "2026-03-14T12:00:00+00:00",
    }


@pytest.fixture
def feature_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair of feature DataFrames for human and synthetic documents."""
    np.random.seed(123)
    n_human, n_synth = 200, 80
    human = pd.DataFrame({
        "doc_id": [f"h-{i}" for i in range(n_human)],
        "perplexity_mean": np.random.normal(45, 10, n_human),
        "watermark_z_score": np.random.normal(0.0, 1.0, n_human),
        "vocabulary_richness": np.random.normal(0.65, 0.08, n_human),
        "sentence_length_std": np.random.normal(8, 3, n_human),
    })
    synthetic = pd.DataFrame({
        "doc_id": [f"s-{i}" for i in range(n_synth)],
        "perplexity_mean": np.random.normal(25, 5, n_synth),
        "watermark_z_score": np.random.normal(2.0, 0.8, n_synth),
        "vocabulary_richness": np.random.normal(0.55, 0.05, n_synth),
        "sentence_length_std": np.random.normal(5, 1.5, n_synth),
    })
    return human, synthetic


# ---------------------------------------------------------------------------
# Tests: curves.py
# ---------------------------------------------------------------------------


class TestTemporalSimilarityCurve:
    """Tests for plot_temporal_similarity_curve."""

    def test_creates_png(self, tmp_path: Path, curve_df: pd.DataFrame) -> None:
        out = tmp_path / "curve.png"
        plot_temporal_similarity_curve(curve_df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_with_inflection(self, tmp_path: Path, curve_df: pd.DataFrame) -> None:
        out = tmp_path / "curve_inflection.png"
        plot_temporal_similarity_curve(curve_df, out, inflection_bin="2020")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_inflection_not_in_bins(self, tmp_path: Path, curve_df: pd.DataFrame) -> None:
        """Non-existent inflection bin should not crash."""
        out = tmp_path / "curve_nobin.png"
        plot_temporal_similarity_curve(curve_df, out, inflection_bin="1999")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path: Path, curve_df: pd.DataFrame) -> None:
        out = tmp_path / "deep" / "nested" / "curve.png"
        plot_temporal_similarity_curve(curve_df, out)
        assert out.exists()


class TestContaminationRate:
    """Tests for plot_contamination_rate."""

    def test_creates_png(self, tmp_path: Path) -> None:
        bins = ["2018", "2019", "2020", "2021", "2022"]
        fracs = [0.02, 0.03, 0.08, 0.15, 0.28]
        out = tmp_path / "contamination.png"
        plot_contamination_rate(bins, fracs, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_zero_contamination(self, tmp_path: Path) -> None:
        """All-zero fractions should still produce a valid chart."""
        bins = ["2020", "2021"]
        fracs = [0.0, 0.0]
        out = tmp_path / "zero.png"
        plot_contamination_rate(bins, fracs, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_bin(self, tmp_path: Path) -> None:
        out = tmp_path / "single.png"
        plot_contamination_rate(["2024"], [0.5], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: distributions.py
# ---------------------------------------------------------------------------


class TestFeatureDistributions:
    """Tests for plot_feature_distributions."""

    def test_creates_png(
        self, tmp_path: Path, feature_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        human, synthetic = feature_dfs
        out = tmp_path / "distributions.png"
        plot_feature_distributions(human, synthetic, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_feature(self, tmp_path: Path) -> None:
        human = pd.DataFrame({"score": np.random.normal(0, 1, 100)})
        synthetic = pd.DataFrame({"score": np.random.normal(1, 1, 50)})
        out = tmp_path / "single_feat.png"
        plot_feature_distributions(human, synthetic, out)
        assert out.exists()

    def test_no_features_skips(self, tmp_path: Path) -> None:
        """DataFrames with only metadata columns should not crash."""
        human = pd.DataFrame({"doc_id": ["a", "b"], "timestamp": [1, 2]})
        synthetic = pd.DataFrame({"doc_id": ["c"], "timestamp": [3]})
        out = tmp_path / "empty.png"
        plot_feature_distributions(human, synthetic, out)
        # No output file since there are no features to plot
        assert not out.exists()


# ---------------------------------------------------------------------------
# Tests: summary.py
# ---------------------------------------------------------------------------


class TestAuditReport:
    """Tests for generate_audit_report."""

    def test_creates_markdown(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
        reserve_summary: dict,
    ) -> None:
        config = {"corpus_name": "Test Corpus", "inflection_bin": "2020"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, reserve_summary, tmp_path,
        )
        assert report_path.exists()
        assert report_path.suffix == ".md"
        assert report_path.stat().st_size > 0

    def test_report_contains_all_sections(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
        reserve_summary: dict,
    ) -> None:
        config = {"corpus_name": "Test Corpus"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, reserve_summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")

        # All seven sections must be present
        assert "## 1. Executive Summary" in content
        assert "## 2. Temporal Similarity Curve" in content
        assert "## 3. Contamination Rate by Year" in content
        assert "## 4. Classifier Performance" in content
        assert "## 5. Feature Importance Ranking" in content
        assert "## 6. Reserve Statistics" in content
        assert "## 7. Recommendations" in content

    def test_report_embeds_image_references(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
        reserve_summary: dict,
    ) -> None:
        config = {"corpus_name": "Image Ref Test"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, reserve_summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "![Temporal Similarity Curve]" in content
        assert "![Contamination Rate]" in content

    def test_report_includes_metrics(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
        reserve_summary: dict,
    ) -> None:
        config = {"corpus_name": "Metrics Test"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, reserve_summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "Accuracy" in content
        assert "AUROC" in content
        assert "0.9650" in content

    def test_report_includes_alpha_t(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
        reserve_summary: dict,
    ) -> None:
        config = {"corpus_name": "Alpha Test"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, reserve_summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "Alpha_t" in content

    def test_report_includes_feature_importance(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
        reserve_summary: dict,
    ) -> None:
        config = {"corpus_name": "Feature Imp Test"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, reserve_summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "perplexity_mean" in content
        assert "watermark_z_score" in content

    def test_supporting_images_created(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
        reserve_summary: dict,
    ) -> None:
        config = {"corpus_name": "Images Test"}
        generate_audit_report(
            config, curve_df, classifier_metrics, reserve_summary, tmp_path,
        )
        assert (tmp_path / "temporal_similarity_curve.png").exists()
        assert (tmp_path / "contamination_rate.png").exists()
        assert (tmp_path / "temporal_similarity_curve.png").stat().st_size > 0
        assert (tmp_path / "contamination_rate.png").stat().st_size > 0

    def test_no_feature_importance(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        reserve_summary: dict,
    ) -> None:
        """Report should handle missing feature_importance gracefully."""
        metrics = {"accuracy": 0.90, "precision": 0.85, "recall": 0.88,
                    "f1": 0.86, "auroc": 0.72, "auprc": 0.80}
        config = {"corpus_name": "No FI Test"}
        report_path = generate_audit_report(
            config, curve_df, metrics, reserve_summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "Feature importance data not available" in content
        # Low AUROC triggers a recommendation
        assert "Classifier confidence is moderate" in content

    def test_report_recommendations_high_contamination(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
    ) -> None:
        """When alpha_t < 0.5, report should warn about high contamination."""
        summary = {
            "total_documents_audited": 1000,
            "reserve_size": 300,
            "alpha_t": 0.30,
            "threshold": 0.90,
            "mean_authenticity_score": 0.45,
            "temporal_distribution": {},
            "source_distribution": {},
        }
        config = {"corpus_name": "High Contamination Test"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "High contamination detected" in content

    def test_report_recommendations_low_contamination(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        classifier_metrics: dict,
    ) -> None:
        """When alpha_t >= 0.8, report should indicate low contamination (line 91)."""
        summary = {
            "total_documents_audited": 1000,
            "reserve_size": 850,
            "alpha_t": 0.85,
            "threshold": 0.90,
            "mean_authenticity_score": 0.92,
            "temporal_distribution": {},
            "source_distribution": {},
        }
        config = {"corpus_name": "Low Contamination Test"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "Low contamination" in content

    def test_report_extra_numeric_metrics(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        reserve_summary: dict,
    ) -> None:
        """Extra numeric metrics not in display_names should appear (line 47)."""
        metrics = {
            "accuracy": 0.90,
            "precision": 0.85,
            "recall": 0.88,
            "f1": 0.86,
            "auroc": 0.95,
            "auprc": 0.91,
            "custom_metric": 0.7777,
        }
        config = {"corpus_name": "Extra Metrics Test"}
        report_path = generate_audit_report(
            config, curve_df, metrics, reserve_summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "custom_metric" in content
        assert "0.7777" in content

    def test_report_feature_importance_as_dataframe(
        self,
        tmp_path: Path,
        curve_df: pd.DataFrame,
        reserve_summary: dict,
    ) -> None:
        """Feature importance passed as a DataFrame should work (line 59)."""
        fi_df = pd.DataFrame([
            {"feature": "ppl_mean", "importance": 0.40},
            {"feature": "wm_z", "importance": 0.30},
        ])
        metrics = {
            "accuracy": 0.90,
            "precision": 0.85,
            "recall": 0.88,
            "f1": 0.86,
            "auroc": 0.95,
            "auprc": 0.91,
            "feature_importance": fi_df,
        }
        config = {"corpus_name": "DF Feature Importance Test"}
        report_path = generate_audit_report(
            config, curve_df, metrics, reserve_summary, tmp_path,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "ppl_mean" in content
        assert "wm_z" in content

    def test_report_zero_docs_in_bin(
        self,
        tmp_path: Path,
        classifier_metrics: dict,
    ) -> None:
        """A bin with n_documents == 0 should produce synthetic_fraction 0.0 (line 183)."""
        curve_df = pd.DataFrame({
            "bin": ["2020", "2021", "2022"],
            "mean_similarity": [0.3, 0.4, 0.5],
            "cross_similarity_to_reference": [0.25, 0.35, 0.45],
            "n_documents": [100, 0, 50],  # 2021 has 0 docs
            "similarity_p25": [0.2, 0.3, 0.4],
            "similarity_p75": [0.4, 0.5, 0.6],
        })
        summary = {
            "total_documents_audited": 150,
            "reserve_size": 100,
            "alpha_t": 0.67,
            "threshold": 0.90,
            "mean_authenticity_score": 0.80,
            "temporal_distribution": {"2020": 80, "2022": 20},
            "source_distribution": {"common_crawl": 100},
        }
        config = {"corpus_name": "Zero Bin Test"}
        report_path = generate_audit_report(
            config, curve_df, classifier_metrics, summary, tmp_path,
        )
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        # Should not crash even with n_documents=0 bin
        assert "Contamination Rate" in content
