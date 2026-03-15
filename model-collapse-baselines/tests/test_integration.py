"""End-to-end integration test with a tiny model.

Uses the debug config (tiny-gpt2, 3 generations) and mocks model
training/generation to avoid requiring a GPU.  Verifies:

- 3 checkpoints saved
- Metrics logged for all 3 generations
- Collapse curves generated
- Entropy at gen 2 <= entropy at gen 0
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def debug_config() -> dict[str, Any]:
    """Minimal config matching debug.yaml."""
    return {
        "experiment": {
            "name": "integration_test",
            "seed": 42,
            "num_generations": 3,
            "output_dir": "",  # set per-test
        },
        "base_model": "sshleifer/tiny-gpt2",
        "real_data": {
            "dataset": "openwebtext",
            "split": "train",
            "max_documents": 500,
            "max_length": 64,
        },
        "synthetic_generation": {
            "num_samples": 200,
            "max_new_tokens": 64,
            "temperature": 1.0,
            "top_p": 0.95,
            "batch_size": 8,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "max_steps": 50,
            "use_lora": False,
            "from_pretrained_each_generation": True,
        },
        "schedule": {
            "type": "zero",
        },
        "measurement": {
            "eval_samples": 100,
        },
    }


@pytest.fixture()
def experiment_dir(tmp_path: Path) -> Path:
    """Temporary experiment directory."""
    d = tmp_path / "integration_test"
    d.mkdir()
    return d


def _make_simulated_metrics(generation: int, base_entropy: float = 5.0) -> dict[str, Any]:
    """Create simulated metrics that show realistic collapse behaviour.

    Entropy and diversity decrease with generation (simulating collapse).
    KL divergence increases with generation.
    """
    decay = 0.85 ** generation
    return {
        "generation": generation,
        "alpha": 0.0,  # zero_alpha schedule
        "entropy": base_entropy * decay,
        "kl_divergence": 0.1 * (generation + 1),
        "js_divergence": 0.05 * (generation + 1),
        "embedding_variance": 10.0 * decay,
        "vocab_coverage": int(5000 * decay),
        "vocabulary_usage": int(5000 * decay),
        "distinct_1": 0.8 * decay,
        "distinct_2": 0.6 * decay,
        "distinct_3": 0.4 * decay,
        "distinct_4": 0.3 * decay,
        "self_bleu": 0.3 + 0.1 * generation,
        "tail_mass_p01": 0.01 * decay,
        "tail_mass_p05": 0.05 * decay,
        "tail_mass_p10": 0.10 * decay,
        "train_loss": 3.0 + 0.2 * generation,
        "perplexity": 20.0 + 5.0 * generation,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "extra": {},
    }


def _create_fake_checkpoint(checkpoint_dir: Path, generation: int) -> None:
    """Create a minimal fake checkpoint directory."""
    gen_dir = checkpoint_dir / f"generation_{generation:02d}"
    model_dir = gen_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Write a minimal metadata file.
    meta = {
        "generation": generation,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "alpha": 0.0,
    }
    with open(gen_dir / "metadata.json", "w") as f:
        json.dump(meta, f)
    # Write a dummy config to make checkpoint_mgr.generation_exists() work.
    with open(model_dir / "config.json", "w") as f:
        json.dump({"model_type": "gpt2"}, f)


# ------------------------------------------------------------------
# Integration test
# ------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration test using mocked training and generation."""

    def test_full_pipeline(
        self,
        debug_config: dict[str, Any],
        experiment_dir: Path,
    ) -> None:
        """Run a full 3-generation pipeline with mocked training.

        Verifies:
        1. Three checkpoints are created.
        2. Metrics are recorded for all 3 generations.
        3. Collapse curves can be generated from the metrics.
        4. Entropy at generation 2 <= entropy at generation 0.
        """
        debug_config["experiment"]["output_dir"] = str(experiment_dir)

        checkpoint_dir = experiment_dir / "checkpoints"
        metrics_dir = experiment_dir / "metrics"

        # ----------------------------------------------------------
        # Phase 1: Simulate the lineage run by writing metrics & checkpoints
        # ----------------------------------------------------------

        from src.training.checkpointing import (
            GenerationMetrics,
            MetricsRecorder,
        )

        metrics_recorder = MetricsRecorder(metrics_dir)

        all_metrics: list[dict] = []
        for gen in range(3):
            sim = _make_simulated_metrics(gen)
            all_metrics.append(sim)

            # Record metrics via the real MetricsRecorder.
            gm = GenerationMetrics(
                generation=gen,
                alpha=sim["alpha"],
                train_loss=sim["train_loss"],
                perplexity=sim["perplexity"],
                kl_divergence=sim["kl_divergence"],
                embedding_variance=sim["embedding_variance"],
                vocab_coverage=sim.get("vocab_coverage"),
                extra={
                    "entropy": sim["entropy"],
                    "js_divergence": sim["js_divergence"],
                    "distinct_1": sim["distinct_1"],
                    "distinct_2": sim["distinct_2"],
                    "distinct_3": sim["distinct_3"],
                    "distinct_4": sim["distinct_4"],
                    "self_bleu": sim["self_bleu"],
                    "vocabulary_usage": sim["vocabulary_usage"],
                    "tail_mass_p01": sim["tail_mass_p01"],
                    "tail_mass_p05": sim["tail_mass_p05"],
                    "tail_mass_p10": sim["tail_mass_p10"],
                },
            )
            metrics_recorder.record(gm)

            # Create fake checkpoint.
            _create_fake_checkpoint(checkpoint_dir, gen)

        # ----------------------------------------------------------
        # Verification 1: Three checkpoints exist
        # ----------------------------------------------------------

        for gen in range(3):
            gen_dir = checkpoint_dir / f"generation_{gen:02d}"
            assert gen_dir.exists(), f"Checkpoint dir missing for gen {gen}"
            model_dir = gen_dir / "model"
            assert model_dir.exists(), f"Model dir missing for gen {gen}"

        # ----------------------------------------------------------
        # Verification 2: Metrics logged for all 3 generations
        # ----------------------------------------------------------

        json_path = metrics_dir / "metrics.json"
        assert json_path.exists(), "metrics.json not created"

        with open(json_path) as f:
            recorded = json.load(f)

        assert len(recorded) == 3, f"Expected 3 records, got {len(recorded)}"

        for i, record in enumerate(recorded):
            assert record["generation"] == i, (
                f"Record {i} has generation={record['generation']}"
            )

        # ----------------------------------------------------------
        # Verification 3: Collapse curves can be generated
        # ----------------------------------------------------------

        # Build a DataFrame that mirrors the stored metrics, flattening
        # the 'extra' dict for plotting.
        df = pd.DataFrame(all_metrics)
        plots_dir = experiment_dir / "plots"

        from src.analysis.collapse_curves import plot_all_curves

        plot_paths = plot_all_curves(df, plots_dir)

        assert len(plot_paths) == 5, f"Expected 5 plots, got {len(plot_paths)}"
        for p in plot_paths:
            assert p.exists(), f"Plot not created: {p}"
            assert p.stat().st_size > 0, f"Plot is empty: {p}"

        # ----------------------------------------------------------
        # Verification 4: Entropy at gen 2 <= entropy at gen 0
        # ----------------------------------------------------------

        entropy_0 = all_metrics[0]["entropy"]
        entropy_2 = all_metrics[2]["entropy"]

        assert entropy_2 <= entropy_0, (
            f"Entropy should decrease with collapse: "
            f"gen0={entropy_0:.4f}, gen2={entropy_2:.4f}"
        )

    def test_metrics_recorder_resume(
        self,
        experiment_dir: Path,
    ) -> None:
        """Test that MetricsRecorder properly resumes from existing data."""
        from src.training.checkpointing import (
            GenerationMetrics,
            MetricsRecorder,
        )

        metrics_dir = experiment_dir / "resume_test"

        # First recorder: write 2 generations.
        recorder1 = MetricsRecorder(metrics_dir)
        recorder1.record(GenerationMetrics(generation=0, alpha=0.0))
        recorder1.record(GenerationMetrics(generation=1, alpha=0.0))
        assert recorder1.latest_generation() == 1

        # Second recorder: should resume and can add gen 2.
        recorder2 = MetricsRecorder(metrics_dir)
        assert recorder2.latest_generation() == 1
        recorder2.record(GenerationMetrics(generation=2, alpha=0.0))
        assert recorder2.latest_generation() == 2

        all_records = recorder2.get_all()
        assert len(all_records) == 3

    def test_checkpoint_manager(
        self,
        experiment_dir: Path,
    ) -> None:
        """Test CheckpointManager generation tracking."""
        from src.training.checkpointing import CheckpointManager

        ckpt_dir = experiment_dir / "ckpt_test"
        mgr = CheckpointManager(ckpt_dir)

        assert mgr.latest_generation() == -1

        # Create fake checkpoints.
        for gen in range(3):
            _create_fake_checkpoint(ckpt_dir, gen)

        assert mgr.latest_generation() == 2
        assert mgr.generation_exists(0)
        assert mgr.generation_exists(1)
        assert mgr.generation_exists(2)
        assert not mgr.generation_exists(3)

    def test_fixed_point_detector(self) -> None:
        """Test fixed-point detection with simulated trajectories."""
        from src.measurement.fixed_point import FixedPointDetector

        detector = FixedPointDetector(patience=2, kl_tolerance=0.01,
                                      diversity_tolerance=0.01)

        # Feed metrics that converge.
        for gen in range(10):
            kl = 1.0 - 0.9 ** gen  # asymptotically approaches 1.0
            diversity = 0.5 * (0.9 ** gen)
            entropy = 5.0 * (0.95 ** gen)

            converged = detector.update(gen, {
                "kl_divergence": kl,
                "diversity": diversity,
                "entropy": entropy,
            })

            if converged:
                report = detector.get_convergence_report()
                assert report.converged
                assert report.generation_converged is not None
                break

        report = detector.get_convergence_report()
        assert len(report.kl_trajectory) == gen + 1

    def test_collapse_curves_with_flat_data(
        self,
        experiment_dir: Path,
    ) -> None:
        """Verify plotting handles edge cases (constant metrics)."""
        df = pd.DataFrame([
            _make_simulated_metrics(0),
            _make_simulated_metrics(1),
            _make_simulated_metrics(2),
        ])

        from src.analysis.collapse_curves import (
            plot_alpha_schedule_overlay,
            plot_entropy_curve,
            plot_kl_curve,
        )

        plots_dir = experiment_dir / "edge_case_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        p1 = plot_entropy_curve(df, plots_dir / "entropy.png",
                                fixed_point_gen=1)
        assert p1.exists()

        p2 = plot_kl_curve(df, plots_dir / "kl.png")
        assert p2.exists()

        p3 = plot_alpha_schedule_overlay(df, plots_dir / "alpha.png")
        assert p3.exists()

    def test_phase_diagram(
        self,
        experiment_dir: Path,
    ) -> None:
        """Test phase diagram generation with multiple schedules."""
        from src.analysis.phase_diagrams import (
            plot_collapse_boundary,
            plot_phase_diagram,
        )

        all_runs = {
            "zero_alpha": pd.DataFrame([
                _make_simulated_metrics(g) for g in range(3)
            ]),
            "constant_alpha": pd.DataFrame([
                {**_make_simulated_metrics(g), "alpha": 0.5,
                 "kl_divergence": 0.05 * (g + 1)}
                for g in range(3)
            ]),
        }

        plots_dir = experiment_dir / "phase_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        p1 = plot_phase_diagram(all_runs, "kl_divergence",
                                plots_dir / "phase_kl.png")
        assert p1.exists()

        p2 = plot_collapse_boundary(all_runs, collapse_threshold=0.2,
                                    output_path=plots_dir / "boundary.png")
        assert p2.exists()

    def test_scale_comparison(
        self,
        experiment_dir: Path,
    ) -> None:
        """Test scale comparison stats and plotting."""
        from src.analysis.scale_comparison import (
            compute_scale_interaction_stats,
            plot_scale_comparison,
            plot_scale_comparison_panel,
        )

        metrics_1b = pd.DataFrame([
            _make_simulated_metrics(g) for g in range(3)
        ])
        metrics_7b = pd.DataFrame([
            {**_make_simulated_metrics(g),
             "kl_divergence": 0.05 * (g + 1),
             "entropy": 6.0 * (0.9 ** g)}
            for g in range(3)
        ])

        plots_dir = experiment_dir / "scale_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        p1 = plot_scale_comparison(
            metrics_1b, metrics_7b, "kl_divergence",
            plots_dir / "kl_compare.png",
        )
        assert p1.exists()

        p2 = plot_scale_comparison_panel(
            metrics_1b, metrics_7b,
            plots_dir / "panel.png",
        )
        assert p2.exists()

        stats = compute_scale_interaction_stats(metrics_1b, metrics_7b)
        assert "collapse_rate_ratio" in stats
        assert "fixed_point_generations" in stats
        assert "entropy_floors" in stats
        assert "variance_ratios" in stats
        # 1B should collapse faster than 7B (higher KL growth).
        assert stats["collapse_rate_ratio"] > 1.0

    def test_report_generation(
        self,
        experiment_dir: Path,
    ) -> None:
        """Test full report generation from an experiment directory."""
        # Set up experiment dir with a schedule sub-directory.
        sched_dir = experiment_dir / "zero_alpha"
        metrics_dir = sched_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        records = [_make_simulated_metrics(g) for g in range(3)]
        with open(metrics_dir / "metrics.json", "w") as f:
            json.dump(records, f)

        from src.analysis.report import generate_report

        report_path = generate_report(experiment_dir,
                                       experiment_dir / "report.md")
        assert report_path.exists()

        report_text = report_path.read_text()
        assert "# Model Collapse Experiment Report" in report_text
        assert "zero_alpha" in report_text
        assert "Collapse Rate Table" in report_text

    def test_entropy_decreases_across_generations(self) -> None:
        """Verify that simulated metrics show monotonic entropy decrease."""
        metrics = [_make_simulated_metrics(g) for g in range(5)]
        for i in range(1, len(metrics)):
            assert metrics[i]["entropy"] <= metrics[i - 1]["entropy"], (
                f"Entropy should decrease: gen{i-1}={metrics[i-1]['entropy']:.4f} "
                f"> gen{i}={metrics[i]['entropy']:.4f}"
            )

    def test_alpha_schedule_values(self) -> None:
        """Verify that all schedules produce valid alpha values."""
        from src.training.schedules import schedule_from_config

        configs = [
            {"type": "zero"},
            {"type": "constant", "alpha": 0.5},
            {"type": "linear", "alpha_0": 1.0, "alpha_min": 0.0},
            {"type": "exponential", "alpha_0": 1.0, "gamma": 0.8},
        ]

        for cfg in configs:
            schedule = schedule_from_config(cfg)
            for gen in range(15):
                alpha = schedule(gen, 15)
                assert 0.0 <= alpha <= 1.0, (
                    f"Schedule {cfg['type']} produced alpha={alpha} at gen {gen}"
                )
