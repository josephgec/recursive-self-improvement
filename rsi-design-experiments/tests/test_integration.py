"""Integration tests: run all 3 experiments end-to-end with reduced parameters."""

import os
import tempfile
import json

import pytest

from src.experiments.modification_frequency import ModificationFrequencyExperiment
from src.experiments.hindsight_target import HindsightTargetExperiment
from src.experiments.rlm_depth import RLMDepthExperiment
from src.experiments.base import ExperimentResult, ConditionResult
from src.harness.runner import ExperimentRunner
from src.harness.controlled_pipeline import MockPipeline, ControlledPipeline
from src.harness.checkpoint_manager import CheckpointManager
from src.harness.parallel_conditions import ParallelConditionRunner
from src.analysis.anova import ANOVAAnalyzer
from src.analysis.recommendation import RecommendationGenerator, PipelineRecommendation
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.optimal_finder import OptimalFinder
from src.analysis.diminishing_returns import DiminishingReturnsAnalyzer
from src.analysis.interaction_effects import InteractionAnalyzer
from src.reporting.experiment_report import generate_experiment_report
from src.reporting.combined_report import generate_combined_report
from src.reporting.config_generator import ConfigGenerator


class TestEndToEnd:
    """End-to-end integration: run all 3 experiments, analyze, recommend."""

    @pytest.fixture
    def all_results(self):
        """Run all 3 experiments with minimal settings."""
        config = {"iterations_per_condition": 3}
        runner = ExperimentRunner(seed=42)

        freq_exp = ModificationFrequencyExperiment(config)
        hint_exp = HindsightTargetExperiment(config)
        depth_exp = RLMDepthExperiment(config)

        results = runner.run_all(
            [freq_exp, hint_exp, depth_exp],
            repetitions=2,
        )
        return results

    def test_all_experiments_run(self, all_results):
        assert len(all_results) == 3

    def test_experiment_names(self, all_results):
        names = [r.experiment_name for r in all_results]
        assert "modification_frequency" in names
        assert "hindsight_target" in names
        assert "rlm_depth" in names

    def test_condition_counts(self, all_results):
        for result in all_results:
            if result.experiment_name == "modification_frequency":
                assert len(result.conditions) == 7
            elif result.experiment_name == "hindsight_target":
                assert len(result.conditions) == 6
            elif result.experiment_name == "rlm_depth":
                assert len(result.conditions) == 7

    def test_repetitions_correct(self, all_results):
        for result in all_results:
            for cond_name, cond_results in result.per_condition_results.items():
                assert len(cond_results) == 2

    def test_anova_runs(self, all_results):
        analyzer = ANOVAAnalyzer()
        for result in all_results:
            scores = result.get_all_scores("composite_score")
            anova_result = analyzer.one_way_anova(scores)
            assert anova_result.f_statistic >= 0
            assert 0 <= anova_result.eta_squared <= 1

    def test_recommendations_generated(self, all_results):
        gen = RecommendationGenerator()
        rec = gen.generate(all_results)
        assert rec.modification_frequency != ""
        assert rec.hindsight_target != ""
        assert isinstance(rec.rlm_depth, int)

    def test_optimal_config_exported(self, all_results):
        gen = RecommendationGenerator()
        rec = gen.generate(all_results)

        config_gen = ConfigGenerator()
        yaml_str = config_gen.generate_optimal_config(rec)
        assert len(yaml_str) > 0
        assert "modification_frequency" in yaml_str
        assert "hindsight_target" in yaml_str
        assert "rlm_depth" in yaml_str

    def test_experiment_reports(self, all_results):
        for result in all_results:
            report = generate_experiment_report(result)
            assert len(report) > 0
            assert result.experiment_name in report
            assert "ANOVA" in report

    def test_combined_report(self, all_results):
        report = generate_combined_report(all_results)
        assert len(report) > 0
        assert "Executive Summary" in report
        assert "Sensitivity Ranking" in report
        assert "Recommended Configuration" in report

    def test_sensitivity_ranking(self, all_results):
        analyzer = SensitivityAnalyzer()
        rankings = analyzer.rank_experiments(all_results)
        assert len(rankings) == 3
        # All should have non-negative sensitivity
        for r in rankings:
            assert r.sensitivity >= 0

    def test_interaction_effects(self, all_results):
        analyzer = InteractionAnalyzer()
        all_scores = {
            r.experiment_name: r.get_all_scores("composite_score")
            for r in all_results
        }
        interactions = analyzer.detect_interactions(all_scores)
        # 3 experiments -> 3 pairs
        assert len(interactions) == 3


class TestCheckpointIntegration:
    """Test checkpoint save/load round-trip."""

    def test_save_and_load(self, sample_experiment_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            filepath = mgr.save(sample_experiment_result)
            assert os.path.exists(filepath)

            loaded = mgr.load(os.path.basename(filepath))
            assert loaded.experiment_name == sample_experiment_result.experiment_name
            assert loaded.conditions == sample_experiment_result.conditions
            assert loaded.repetitions == sample_experiment_result.repetitions

            for cond_name in loaded.conditions:
                orig = sample_experiment_result.per_condition_results[cond_name]
                reloaded = loaded.per_condition_results[cond_name]
                assert len(orig) == len(reloaded)
                for o, r in zip(orig, reloaded):
                    assert o.final_accuracy == pytest.approx(r.final_accuracy)
                    assert o.composite_score == pytest.approx(r.composite_score)

    def test_resume_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            result = mgr.resume("nonexistent")
            assert result is None

    def test_resume_exists(self, sample_experiment_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save(sample_experiment_result)
            loaded = mgr.resume("test_experiment")
            assert loaded is not None
            assert loaded.experiment_name == "test_experiment"

    def test_list_checkpoints(self, sample_experiment_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save(sample_experiment_result)
            checkpoints = mgr.list()
            assert len(checkpoints) == 1
            assert "test_experiment_checkpoint.json" in checkpoints


class TestControlledPipelineIntegration:
    """Test ControlledPipeline with different configurations."""

    def test_run_with_frequency_config(self):
        from src.conditions.frequency_conditions import ModificationFrequencyPolicy

        pipeline = ControlledPipeline(seed=42)
        policy = ModificationFrequencyPolicy("every_n", param=5)
        results = pipeline.run_with_config(
            {"modification_policy": policy, "condition_name": "every_5", "seed": 42},
            iterations=10,
        )
        assert len(results) == 10
        for r in results:
            assert "accuracy" in r
            assert "rollback" in r

    def test_run_with_depth_config(self):
        pipeline = ControlledPipeline(seed=42)
        results = pipeline.run_with_config(
            {"depth": 3, "condition_name": "depth_3", "seed": 42},
            iterations=10,
        )
        assert len(results) == 10

    def test_deterministic_with_same_seed(self):
        pipeline1 = ControlledPipeline(seed=42)
        pipeline2 = ControlledPipeline(seed=42)

        results1 = pipeline1.run_with_config(
            {"depth": 3, "condition_name": "depth_3", "seed": 42},
            iterations=5,
        )
        results2 = pipeline2.run_with_config(
            {"depth": 3, "condition_name": "depth_3", "seed": 42},
            iterations=5,
        )
        for r1, r2 in zip(results1, results2):
            assert r1["accuracy"] == pytest.approx(r2["accuracy"])


class TestParallelConditionRunnerIntegration:
    """Test parallel condition runner."""

    def test_run_parallel_conditions(self):
        runner = ParallelConditionRunner(max_workers=2)

        def make_func(name, score):
            def func():
                return ConditionResult(condition_name=name, composite_score=score)
            return func

        condition_funcs = [
            ("cond_a", make_func("cond_a", 0.8)),
            ("cond_b", make_func("cond_b", 0.7)),
            ("cond_c", make_func("cond_c", 0.6)),
        ]

        results = runner.run(condition_funcs)
        assert len(results) == 3
        assert results["cond_a"].composite_score == pytest.approx(0.8)
        assert results["cond_b"].composite_score == pytest.approx(0.7)

    def test_run_repetitions(self):
        runner = ParallelConditionRunner(max_workers=2)

        def func(rep_idx):
            return ConditionResult(
                condition_name=f"rep_{rep_idx}",
                composite_score=0.5 + rep_idx * 0.1,
            )

        results = runner.run_repetitions(func, 3)
        assert len(results) == 3
        assert results[0].composite_score == pytest.approx(0.5)
        assert results[1].composite_score == pytest.approx(0.6)
        assert results[2].composite_score == pytest.approx(0.7)

    def test_parallel_handles_errors(self):
        runner = ParallelConditionRunner(max_workers=2)

        def failing_func():
            raise ValueError("intentional error")

        condition_funcs = [
            ("cond_ok", lambda: ConditionResult(condition_name="cond_ok", composite_score=0.8)),
            ("cond_fail", failing_func),
        ]

        results = runner.run(condition_funcs)
        assert len(results) == 2
        assert results["cond_ok"].composite_score == pytest.approx(0.8)
        assert "error" in results["cond_fail"].metadata


class TestExperimentResultMethods:
    """Test ExperimentResult helper methods."""

    def test_get_mean_accuracy(self, sample_experiment_result):
        mean_a = sample_experiment_result.get_mean_accuracy("cond_a")
        assert mean_a == pytest.approx(0.85, abs=0.03)

    def test_get_mean_accuracy_missing(self, sample_experiment_result):
        mean = sample_experiment_result.get_mean_accuracy("nonexistent")
        assert mean == 0.0

    def test_get_mean_composite(self, sample_experiment_result):
        mean = sample_experiment_result.get_mean_composite("cond_a")
        assert mean > 0

    def test_get_mean_composite_missing(self, sample_experiment_result):
        mean = sample_experiment_result.get_mean_composite("nonexistent")
        assert mean == 0.0

    def test_get_all_scores(self, sample_experiment_result):
        scores = sample_experiment_result.get_all_scores("composite_score")
        assert "cond_a" in scores
        assert len(scores["cond_a"]) == 3


class TestReportWithSignificantAnova:
    """Test experiment report when ANOVA is significant (covers Tukey HSD output)."""

    def test_report_includes_tukey_hsd(self, sample_experiment_result):
        """sample_experiment_result has clearly different condition scores,
        so ANOVA should be significant and Tukey HSD should appear."""
        report = generate_experiment_report(sample_experiment_result)
        assert "Tukey HSD" in report
        # Should contain pairwise comparisons
        assert "vs" in report

    def test_report_with_empty_condition(self):
        """Condition with no results should be skipped gracefully."""
        result = ExperimentResult(
            experiment_name="test_empty",
            conditions=["cond_a", "cond_empty"],
            repetitions=2,
        )
        result.per_condition_results = {
            "cond_a": [
                ConditionResult(condition_name="cond_a", composite_score=0.8,
                                final_accuracy=0.8, stability_score=0.9,
                                total_cost=1.0, accuracy_trajectory=[0.6, 0.8]),
                ConditionResult(condition_name="cond_a", composite_score=0.82,
                                final_accuracy=0.82, stability_score=0.88,
                                total_cost=1.1, accuracy_trajectory=[0.6, 0.82]),
            ],
            "cond_empty": [],
        }
        report = generate_experiment_report(result)
        assert "test_empty" in report


class TestConfigGenerator:
    """Test config generator."""

    def test_generate_optimal_config(self):
        rec = PipelineRecommendation(
            modification_frequency="every_5",
            hindsight_target="both",
            rlm_depth=3,
            confidence_levels={
                "modification_frequency": "high",
                "hindsight_target": "medium",
                "rlm_depth": "low",
            },
            reasoning={"modification_frequency": "Best overall"},
        )
        gen = ConfigGenerator()
        yaml_str = gen.generate_optimal_config(rec)
        assert "every_5" in yaml_str
        assert "both" in yaml_str
        assert "3" in yaml_str
        assert "high" in yaml_str

    def test_generate_minimal_config(self):
        rec = PipelineRecommendation(
            modification_frequency="every_5",
            hindsight_target="both",
            rlm_depth=3,
        )
        gen = ConfigGenerator()
        yaml_str = gen.generate_minimal_config(rec)
        assert "every_5" in yaml_str
        assert "both" in yaml_str


class TestMockPipelineBehavior:
    """Test that MockPipeline produces expected behavior patterns."""

    def test_frequency_every_task_more_rollbacks(self):
        """every_task should have more rollbacks than never."""
        from src.conditions.frequency_conditions import ModificationFrequencyPolicy

        pipeline = MockPipeline(seed=42)
        pipeline.set_condition_name("every_task")
        pipeline.set_modification_policy(ModificationFrequencyPolicy("every_task"))

        rollbacks_et = sum(
            1 for i in range(100) if pipeline.step(i).get("rollback", False)
        )

        pipeline2 = MockPipeline(seed=42)
        pipeline2.set_condition_name("never")
        pipeline2.set_modification_policy(ModificationFrequencyPolicy("never"))

        rollbacks_never = sum(
            1 for i in range(100) if pipeline2.step(i).get("rollback", False)
        )

        assert rollbacks_et > rollbacks_never

    def test_depth_accuracy_model(self):
        """Higher depth should give higher accuracy in step results."""
        import math

        pipeline_low = MockPipeline(seed=42)
        pipeline_low.set_condition_name("depth_0")
        pipeline_low.set_depth(0)

        pipeline_high = MockPipeline(seed=42)
        pipeline_high.set_condition_name("depth_6")
        pipeline_high.set_depth(6)

        # Run many steps and average
        accs_low = [pipeline_low.step(i)["accuracy"] for i in range(15, 50)]
        accs_high = [pipeline_high.step(i)["accuracy"] for i in range(15, 50)]

        mean_low = sum(accs_low) / len(accs_low)
        mean_high = sum(accs_high) / len(accs_high)
        assert mean_high > mean_low

    def test_hindsight_both_better_than_none(self):
        """'both' hindsight target should produce higher accuracy than 'none'."""
        from src.conditions.hindsight_conditions import HindsightTargetPolicy

        pipeline_both = MockPipeline(seed=42)
        pipeline_both.set_condition_name("both")
        pipeline_both.set_hindsight_policy(HindsightTargetPolicy("both"))

        pipeline_none = MockPipeline(seed=42)
        pipeline_none.set_condition_name("none")
        pipeline_none.set_hindsight_policy(HindsightTargetPolicy("none"))

        accs_both = [pipeline_both.step(i)["accuracy"] for i in range(15, 50)]
        accs_none = [pipeline_none.step(i)["accuracy"] for i in range(15, 50)]

        mean_both = sum(accs_both) / len(accs_both)
        mean_none = sum(accs_none) / len(accs_none)
        assert mean_both > mean_none

    def test_pipeline_set_seed_resets(self):
        """Setting seed should reset internal state."""
        pipeline = MockPipeline(seed=42)
        pipeline.set_condition_name("depth_3")
        pipeline.set_depth(3)
        r1 = pipeline.step(0)

        pipeline.set_seed(42)
        r2 = pipeline.step(0)
        assert r1["accuracy"] == pytest.approx(r2["accuracy"])


class TestExperimentRunnerIntegration:
    """Test ExperimentRunner functionality."""

    def test_run_and_get_results(self):
        config = {"iterations_per_condition": 3}
        runner = ExperimentRunner(seed=42)

        exp = ModificationFrequencyExperiment(config)
        result = runner.run_experiment(exp, repetitions=2)

        assert result.experiment_name == "modification_frequency"
        stored = runner.get_results()
        assert "modification_frequency" in stored

    def test_get_result_specific(self):
        config = {"iterations_per_condition": 3}
        runner = ExperimentRunner(seed=42)

        exp = HindsightTargetExperiment(config)
        runner.run_experiment(exp, repetitions=2)

        result = runner.get_result("hindsight_target")
        assert result is not None
        assert result.experiment_name == "hindsight_target"

    def test_get_result_missing(self):
        runner = ExperimentRunner(seed=42)
        assert runner.get_result("nonexistent") is None

    def test_run_all(self):
        config = {"iterations_per_condition": 3}
        runner = ExperimentRunner(seed=42)

        experiments = [
            ModificationFrequencyExperiment(config),
            HindsightTargetExperiment(config),
        ]
        results = runner.run_all(experiments, repetitions=2)
        assert len(results) == 2
        stored = runner.get_results()
        assert len(stored) == 2
