"""Integration tests: full pipeline from training to safety report."""

import numpy as np
import pytest

from src.eppo.config import EPPOConfig
from src.eppo.trainer import EPPOTrainer
from src.eppo.entropy_bonus import EntropyBonus
from src.eppo.policy import MockPolicy
from src.eppo.value_head import MockValueHead
from src.bounding.process_reward import ProcessRewardShaper
from src.bounding.reward_clipper import RewardClipper
from src.bounding.delta_bounder import DeltaBounder
from src.bounding.reward_normalizer import RewardNormalizer
from src.bounding.reward_monitor import RewardMonitor
from src.energy.energy_tracker import EnergyTracker
from src.energy.homogenization import HomogenizationDetector
from src.energy.early_warning import EnergyEarlyWarning
from src.energy.layer_norms import LayerNormTracker
from src.detection.composite_detector import (
    CompositeRewardHackingDetector,
    TrainingState,
)
from src.deliverables.phase_gate import PhaseGateSafetyPackage
from src.deliverables.safety_report import generate_safety_report
from src.integration.soar_adapter import SOARRewardHackingAdapter
from src.integration.pipeline_adapter import PipelineRewardAdapter
from src.integration.training_wrapper import MitigatedTrainingWrapper
from src.analysis.report import generate_full_report
from src.analysis.eppo_analysis import analyze_eppo_training
from src.analysis.bounding_analysis import analyze_bounding
from src.analysis.energy_analysis import analyze_energy


class TestFullPipeline:
    """Test the full pipeline: training -> bounding -> energy -> detection -> package -> report."""

    def test_end_to_end(self):
        """Full end-to-end pipeline test."""
        rng = np.random.RandomState(42)

        # 1. EPPO Training
        config = EPPOConfig(
            entropy_mode="coefficient",
            entropy_coeff=0.01,
            decay_rate=0.99,
            min_beta=0.001,
            hidden_dim=32,
            vocab_size=50,
            batch_size=16,
        )
        trainer = EPPOTrainer(config)

        for epoch in range(3):
            result = trainer.train_epoch(num_steps=5)
            assert result.mean_entropy > 0

        # 2. Reward Bounding
        shaper = ProcessRewardShaper(
            clip_min=-3.0,
            clip_max=3.0,
            delta_max=1.5,
            normalize=False,  # Skip normalization so extremes get clipped
        )

        rewards = rng.randn(30) * 5  # Some will be clipped (>3 or <-3)
        shaped_results = shaper.shape_batch(rewards)
        assert len(shaped_results) == 30

        # Check bounding is working
        clipped_count = sum(1 for sr in shaped_results if sr.was_clipped)
        assert clipped_count > 0  # Some should be clipped

        # 3. Energy Tracking
        energy_tracker = EnergyTracker(num_layers=4)
        for step in range(20):
            scale = max(0.1, 1.0 - 0.01 * step)
            activations = [rng.randn(32) * scale for _ in range(4)]
            energy_tracker.measure(activations)

        if len(energy_tracker.measurements) >= 5:
            energy_tracker.set_baseline(
                energy=energy_tracker.measurements[0].total_energy
            )

        # 4. Homogenization Detection
        homog_detector = HomogenizationDetector()
        homog_result = homog_detector.detect(energy_tracker.measurements)
        assert homog_result is not None

        # 5. Early Warning
        warning = EnergyEarlyWarning()
        pred = warning.predict(energy_tracker.measurements, horizon=10)
        assert pred is not None

        # 6. Detection
        detector = CompositeRewardHackingDetector(divergence_window=10)
        state = TrainingState(
            rewards=[sr.final for sr in shaped_results[:15]],
            accuracies=[0.5 + rng.randn() * 0.05 for _ in range(15)],
            output_lengths=[50 + int(rng.randn() * 5) for _ in range(10)],
            baseline_lengths=[48 + int(rng.randn() * 5) for _ in range(10)],
            outputs=[list(rng.randint(0, 100, 50)) for _ in range(10)],
            output_strings=["Normal output text"] * 10,
        )
        report = detector.check(state)
        assert report is not None

        # 7. Safety Package
        gate = PhaseGateSafetyPackage()
        histories = {
            "gdi": {},
            "constraint": {
                "reward_bounded": True,
                "entropy_above_min": trainer.entropy_bonus.current_beta > config.min_beta,
                "energy_stable": not energy_tracker.is_declining(),
            },
            "interp": {
                "energy_interpretable": True,
                "homogenization_checked": True,
            },
            "reward": {
                "no_divergence": not report.is_hacking_detected,
                "no_shortcuts": True,
                "no_gaming": True,
            },
        }
        pkg = gate.package("phase_1", (0, 15), histories)
        validation = gate.validate(pkg)

        # 8. Safety Report
        safety_report = generate_safety_report(pkg)
        assert "# Safety Report" in safety_report
        assert "phase_1" in safety_report

        # 9. Analysis Report
        analysis_report = generate_full_report(
            epoch_results=trainer.epoch_results,
            shaped_rewards=shaper.history,
            energy_measurements=energy_tracker.measurements,
        )
        assert "EPPO Training Analysis" in analysis_report
        assert "Reward Bounding Analysis" in analysis_report
        assert "Energy Analysis" in analysis_report

    def test_soar_adapter(self):
        """Test SOAR adapter wraps training correctly."""
        adapter = SOARRewardHackingAdapter()
        result = adapter.wrap_training(num_steps=20, check_interval=10)

        assert result.total_steps == 20
        assert not result.was_stopped_early
        assert result.final_report is not None

    def test_soar_step(self):
        """Test individual SOAR step."""
        adapter = SOARRewardHackingAdapter()
        batch = adapter.trainer._make_random_batch()
        raw_rewards = batch["rewards"].copy()

        step_result = adapter.on_training_step(batch, raw_rewards)
        assert step_result.step == 0
        assert step_result.energy is not None

    def test_soar_epoch_end(self):
        """Test SOAR epoch end check."""
        adapter = SOARRewardHackingAdapter()
        # Run a few steps first
        for _ in range(5):
            batch = adapter.trainer._make_random_batch()
            adapter.on_training_step(batch, batch["rewards"].copy())

        report = adapter.on_epoch_end()
        assert report is not None

    def test_pipeline_adapter(self):
        """Test pipeline reward adapter."""
        adapter = PipelineRewardAdapter()
        rng = np.random.RandomState(42)

        for _ in range(10):
            result = adapter.process(rng.randn(10))
            assert len(result.shaped_rewards) == 10

        assert len(adapter.results) == 10

    def test_pipeline_reset(self):
        """Test pipeline adapter reset."""
        adapter = PipelineRewardAdapter()
        adapter.process(np.array([1.0, 2.0, 3.0]))
        adapter.reset()
        assert len(adapter.results) == 0

    def test_mitigated_training_wrapper(self):
        """Test mitigated training wrapper."""
        wrapper = MitigatedTrainingWrapper()
        result = wrapper.run_full_training(num_epochs=2, steps_per_epoch=5)

        assert result.total_steps == 10
        assert result.epochs_completed == 2
        assert result.mitigation_status is not None

    def test_mitigated_before_training(self):
        """Test before_training initializes defenses."""
        wrapper = MitigatedTrainingWrapper()
        status = wrapper.before_training()
        assert status.eppo_active
        assert status.bounding_active
        assert status.energy_monitoring
        assert status.detection_active

    def test_mitigated_after_step(self):
        """Test after_step runs checks."""
        wrapper = MitigatedTrainingWrapper()
        wrapper.before_training()
        batch = wrapper.trainer._make_random_batch()
        status = wrapper.after_step(0, batch)
        assert isinstance(status.warnings, list)

    def test_mitigated_after_training(self):
        """Test after_training returns summary."""
        wrapper = MitigatedTrainingWrapper()
        wrapper.before_training()
        wrapper.trainer.train_epoch(num_steps=5)
        result = wrapper.after_training()
        assert result.total_steps == 5


class TestAnalysis:
    """Test analysis modules."""

    def test_eppo_analysis(self):
        """analyze_eppo_training works on real data."""
        trainer = EPPOTrainer(EPPOConfig(hidden_dim=32, vocab_size=50, batch_size=8))
        for _ in range(3):
            trainer.train_epoch(num_steps=5)

        result = analyze_eppo_training(trainer.epoch_results)
        assert result["num_epochs"] == 3
        assert "entropy" in result
        assert "loss" in result
        assert "beta" in result

    def test_eppo_analysis_empty(self):
        """analyze_eppo_training handles empty data."""
        result = analyze_eppo_training([])
        assert result["status"] == "no_data"

    def test_bounding_analysis(self):
        """analyze_bounding works on real data."""
        shaper = ProcessRewardShaper()
        rng = np.random.RandomState(42)
        for _ in range(20):
            shaper.shape(rng.randn() * 5)

        result = analyze_bounding(shaper.history)
        assert result["num_rewards"] == 20
        assert "raw" in result
        assert "shaped" in result

    def test_bounding_analysis_empty(self):
        """analyze_bounding handles empty data."""
        result = analyze_bounding([])
        assert result["status"] == "no_data"

    def test_energy_analysis(self):
        """analyze_energy works on real data."""
        tracker = EnergyTracker(num_layers=4)
        rng = np.random.RandomState(42)
        for _ in range(20):
            tracker.measure([rng.randn(32) for _ in range(4)])

        result = analyze_energy(tracker.measurements)
        assert result["num_measurements"] == 20
        assert "total_energy" in result
        assert "per_layer" in result

    def test_energy_analysis_empty(self):
        """analyze_energy handles empty data."""
        result = analyze_energy([])
        assert result["status"] == "no_data"


class TestMockComponents:
    """Test mock model components in detail."""

    def test_mock_policy_forward(self):
        """MockPolicy forward produces correct shapes."""
        policy = MockPolicy(input_dim=32, vocab_size=50)

        # Single input
        x = np.random.randn(32)
        logits = policy.forward(x)
        assert logits.shape == (50,)

        # Batch input
        x = np.random.randn(8, 32)
        logits = policy.forward(x)
        assert logits.shape == (8, 50)

    def test_mock_policy_entropy(self):
        """MockPolicy computes entropy correctly."""
        policy = MockPolicy(input_dim=32, vocab_size=50)
        logits = np.random.randn(50)
        entropy = policy.compute_entropy(logits)
        assert entropy > 0
        assert len(policy.entropy_history) == 1

    def test_mock_policy_log_probs(self):
        """MockPolicy computes valid log probabilities."""
        policy = MockPolicy(input_dim=32, vocab_size=50)
        logits = np.random.randn(8, 50)
        log_probs = policy.get_log_probs(logits)
        assert log_probs.shape == (8, 50)
        assert np.all(log_probs <= 0)  # Log probs are non-positive

    def test_mock_policy_full_forward(self):
        """MockPolicy forward_full returns complete output."""
        policy = MockPolicy(input_dim=32, vocab_size=50)
        x = np.random.randn(32)
        output = policy.forward_full(x)
        assert output.logits.shape == (50,)
        assert output.entropy > 0
        assert output.log_probs.shape == (1, 50)

    def test_mock_policy_update(self):
        """MockPolicy update_weights modifies weights."""
        policy = MockPolicy(input_dim=32, vocab_size=50)
        old_weights = policy._weights.copy()
        gradient = np.random.randn(32, 50)
        policy.update_weights(gradient, lr=0.1)
        assert not np.allclose(policy._weights, old_weights)

    def test_mock_policy_perturb(self):
        """MockPolicy perturb_weights adds noise."""
        policy = MockPolicy(input_dim=32, vocab_size=50)
        old_weights = policy._weights.copy()
        policy.perturb_weights(scale=0.1)
        assert not np.allclose(policy._weights, old_weights)

    def test_mock_value_head_predict(self):
        """MockValueHead predicts scalar values."""
        vh = MockValueHead(input_dim=32)
        state = np.random.randn(32)
        value = vh.predict_value(state)
        assert isinstance(value, float)
        assert len(vh.prediction_history) == 1

    def test_mock_value_head_batch(self):
        """MockValueHead predicts batch values."""
        vh = MockValueHead(input_dim=32)
        states = np.random.randn(8, 32)
        values = vh.predict_batch(states)
        assert values.shape == (8,)

    def test_mock_value_head_drift(self):
        """MockValueHead detects drift."""
        vh = MockValueHead(input_dim=32)
        # Add stable predictions
        for _ in range(50):
            vh.predict_value(np.random.randn(32))

        result = vh.detect_drift(window=20)
        assert result.trend in ("stable", "increasing", "decreasing")

    def test_mock_value_head_insufficient_data(self):
        """MockValueHead handles insufficient data for drift."""
        vh = MockValueHead(input_dim=32)
        vh.predict_value(np.random.randn(32))
        result = vh.detect_drift(window=20)
        assert not result.is_drifting

    def test_reward_normalizer(self):
        """RewardNormalizer normalizes correctly."""
        normalizer = RewardNormalizer(window=50)

        # Feed many values to establish stats
        rng = np.random.RandomState(42)
        for _ in range(100):
            normalizer.normalize(rng.randn() * 5 + 10)

        # After many samples, mean should be close to 10, std close to 5
        assert abs(normalizer.mean - 10) < 2.0

    def test_reward_normalizer_batch(self):
        """RewardNormalizer handles batches."""
        normalizer = RewardNormalizer()
        result = normalizer.normalize_batch(np.array([1.0, 2.0, 3.0]))
        assert len(result) == 3

    def test_reward_normalizer_reset(self):
        """RewardNormalizer reset clears state."""
        normalizer = RewardNormalizer()
        normalizer.normalize(5.0)
        normalizer.reset()
        assert normalizer.count == 0

    def test_reward_monitor(self):
        """RewardMonitor tracks and detects anomalies."""
        monitor = RewardMonitor(window=20, z_threshold=2.0)
        rng = np.random.RandomState(42)

        # Normal rewards
        for _ in range(100):
            monitor.record(rng.randn())

        snapshot = monitor.take_snapshot()
        assert snapshot is not None

        anomaly = monitor.detect_anomaly()
        assert not anomaly.is_anomalous

    def test_reward_monitor_anomaly(self):
        """RewardMonitor detects anomalous shift."""
        monitor = RewardMonitor(window=20, z_threshold=2.0)

        # Normal phase
        for _ in range(60):
            monitor.record(0.0 + np.random.randn() * 0.1)

        # Anomalous phase
        for _ in range(20):
            monitor.record(10.0 + np.random.randn() * 0.1)

        anomaly = monitor.detect_anomaly()
        assert anomaly.is_anomalous

    def test_reward_monitor_trend(self):
        """RewardMonitor computes trends."""
        monitor = RewardMonitor(window=10)

        # Insufficient data
        for _ in range(5):
            monitor.record(1.0)
        assert monitor.get_trend() == "insufficient_data"

    def test_process_reward_shaper(self):
        """ProcessRewardShaper applies full pipeline."""
        shaper = ProcessRewardShaper(
            clip_min=-2.0,
            clip_max=2.0,
            delta_max=1.0,
        )

        result = shaper.shape(0.5)
        assert result.raw == 0.5
        assert result.final == result.bounded

    def test_process_reward_shaper_batch(self):
        """ProcessRewardShaper handles batches."""
        shaper = ProcessRewardShaper()
        results = shaper.shape_batch(np.array([1.0, 2.0, 3.0]))
        assert len(results) == 3

    def test_process_reward_shaper_reset(self):
        """ProcessRewardShaper reset clears state."""
        shaper = ProcessRewardShaper()
        shaper.shape(1.0)
        shaper.reset()
        assert len(shaper.history) == 0

    def test_process_reward_no_normalize(self):
        """ProcessRewardShaper works without normalization."""
        shaper = ProcessRewardShaper(normalize=False)
        result = shaper.shape(1.5)
        assert result.normalized == 1.5  # No normalization
