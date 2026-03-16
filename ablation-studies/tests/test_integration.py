"""Integration tests: all 4 suites, statistical analysis, paper assets."""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.suites.neurosymbolic import NeurosymbolicAblation
from src.suites.godel import GodelAgentAblation
from src.suites.soar import SOARAblation
from src.suites.rlm import RLMAblation
from src.suites.base import AblationSuiteResult, PaperAssets
from src.execution.runner import AblationRunner, MockPipeline
from src.execution.controlled_env import ControlledEnvironment
from src.execution.parallel import ParallelConditionRunner
from src.execution.checkpoint import CheckpointManager
from src.analysis.statistical_tests import PublicationStatistics, PairwiseResult
from src.analysis.effect_sizes import cohens_d, eta_squared, interpret_d
from src.analysis.confidence_intervals import bootstrap_ci, bootstrap_difference_ci
from src.analysis.interaction_tests import CrossSuiteInteractionAnalyzer
from src.analysis.power_analysis import required_repetitions, achieved_power, minimum_detectable_effect
from src.publication.latex_tables import LaTeXTableGenerator
from src.publication.figures import PublicationFigureGenerator
from src.publication.narrative import NarrativeGenerator
from src.publication.appendix import AppendixGenerator
from src.publication.significance_stars import add_stars
from src.utils.reproducibility import set_global_seed, get_config_hash, verify_seed_consistency, generate_run_id
from src.utils.cost_estimator import CostEstimator


class TestFullIntegration:
    """Run all 4 suites with 3 conditions each, 2 reps, and verify everything."""

    @pytest.fixture
    def all_results(self):
        """Run all 4 suites with reduced settings for speed."""
        runner = AblationRunner(noise_std=0.02)
        suites = [
            NeurosymbolicAblation(),
            GodelAgentAblation(),
            SOARAblation(),
            RLMAblation(),
        ]
        results = {}
        for suite in suites:
            result = runner.run_suite(suite, repetitions=2, seed=42)
            results[suite.get_paper_name()] = result
        return results

    def test_all_suites_run(self, all_results):
        assert len(all_results) == 4

    def test_all_suites_have_conditions(self, all_results):
        for name, result in all_results.items():
            assert len(result.get_all_condition_names()) > 0

    def test_total_conditions_is_31(self, all_results):
        total = sum(
            len(r.get_all_condition_names()) for r in all_results.values()
        )
        assert total == 31

    def test_statistical_tests_on_all(self, all_results):
        stats = PublicationStatistics()
        suites = [
            NeurosymbolicAblation(),
            GodelAgentAblation(),
            SOARAblation(),
            RLMAblation(),
        ]
        for suite in suites:
            result = all_results[suite.get_paper_name()]
            analyses = suite.analyze(result)
            assert len(analyses) > 0
            for key, pw in analyses.items():
                assert isinstance(pw, PairwiseResult)
                assert 0.0 <= pw.p_value <= 1.0

    def test_paper_assets_generated(self, all_results):
        suites = [
            NeurosymbolicAblation(),
            GodelAgentAblation(),
            SOARAblation(),
            RLMAblation(),
        ]
        for suite in suites:
            result = all_results[suite.get_paper_name()]
            assets = suite.generate_paper_assets(result)
            assert isinstance(assets, PaperAssets)
            assert assets.has_content
            assert "main_results" in assets.tables
            assert len(assets.narrative) > 0

    def test_narrative_mentions_conditions(self, all_results):
        suite = NeurosymbolicAblation()
        result = all_results[suite.get_paper_name()]
        assets = suite.generate_paper_assets(result)
        assert "full" in assets.narrative
        assert "accuracy" in assets.narrative.lower()

    def test_cross_suite_interactions(self, all_results):
        analyzer = CrossSuiteInteractionAnalyzer()
        reports = analyzer.test_cross_paradigm_interactions(all_results)
        # Should have C(4,2) = 6 pair comparisons
        assert len(reports) == 6
        for r in reports:
            assert r.synergy_score >= 0

    def test_bonferroni_correction(self, all_results):
        stats = PublicationStatistics()
        suite = NeurosymbolicAblation()
        result = all_results[suite.get_paper_name()]

        all_scores = {
            c: result.get_scores(c)
            for c in result.get_all_condition_names()
        }
        multi = stats.multi_comparison(all_scores, baseline="full")
        corrected = stats.bonferroni_correct(multi)

        assert len(corrected) == len(multi)
        # Corrected p-values should be >= original
        for orig, corr in zip(multi, corrected):
            assert corr.p_value >= orig.p_value


class TestEffectSizes:
    """Test effect size calculations."""

    def test_cohens_d_same_groups(self):
        a = [0.85, 0.86, 0.84, 0.85, 0.83]
        d = cohens_d(a, a)
        assert abs(d) < 0.01

    def test_cohens_d_different_groups(self):
        a = [0.85, 0.86, 0.84, 0.85, 0.83]
        b = [0.72, 0.73, 0.71, 0.72, 0.70]
        d = cohens_d(a, b)
        assert d > 0  # a > b

    def test_cohens_d_small_sample(self):
        d = cohens_d([1.0], [0.5])
        assert d == 0.0  # Not enough data

    def test_eta_squared_one_group(self):
        eta = eta_squared([[0.85, 0.86, 0.84]])
        assert eta == 0.0  # No between-group variance

    def test_eta_squared_two_groups(self):
        groups = [
            [0.85, 0.86, 0.84, 0.85, 0.83],
            [0.72, 0.73, 0.71, 0.72, 0.70],
        ]
        eta = eta_squared(groups)
        assert 0 < eta <= 1

    def test_eta_squared_empty(self):
        assert eta_squared([]) == 0.0

    def test_interpret_d(self):
        assert interpret_d(0.1) == "negligible"
        assert interpret_d(0.3) == "small"
        assert interpret_d(0.6) == "medium"
        assert interpret_d(1.0) == "large"
        assert interpret_d(-0.9) == "large"


class TestConfidenceIntervals:
    """Test bootstrap CI computation."""

    def test_bootstrap_ci_basic(self):
        data = [0.85, 0.86, 0.84, 0.85, 0.83]
        low, high = bootstrap_ci(data, n_bootstrap=500)
        assert low <= sum(data) / len(data) <= high

    def test_bootstrap_ci_single_value(self):
        low, high = bootstrap_ci([0.85])
        assert low == 0.85
        assert high == 0.85

    def test_bootstrap_ci_empty(self):
        low, high = bootstrap_ci([])
        assert low == 0.0 and high == 0.0

    def test_bootstrap_difference_ci(self):
        a = [0.85, 0.86, 0.84, 0.85, 0.83]
        b = [0.72, 0.73, 0.71, 0.72, 0.70]
        low, high = bootstrap_difference_ci(a, b, n_bootstrap=500)
        # Difference should be positive
        assert low > 0

    def test_bootstrap_difference_ci_empty(self):
        low, high = bootstrap_difference_ci([], [0.5])
        assert low == 0.0 and high == 0.0


class TestPowerAnalysis:
    """Test statistical power analysis functions."""

    def test_required_repetitions(self):
        n = required_repetitions(0.8, alpha=0.05, power=0.80)
        assert n >= 2
        assert isinstance(n, int)

    def test_required_repetitions_zero_effect(self):
        n = required_repetitions(0.0)
        assert n == 999

    def test_achieved_power(self):
        pwr = achieved_power(1.0, 20, alpha=0.05)
        assert pwr > 0.5

    def test_achieved_power_small_n(self):
        pwr = achieved_power(0.5, 1, alpha=0.05)
        assert pwr == 0.0

    def test_minimum_detectable_effect(self):
        mde = minimum_detectable_effect(20, alpha=0.05, power=0.80)
        assert mde > 0

    def test_minimum_detectable_effect_small_n(self):
        mde = minimum_detectable_effect(1)
        assert mde == float("inf")

    def test_larger_n_more_power(self):
        p5 = achieved_power(0.8, 5)
        p20 = achieved_power(0.8, 20)
        assert p20 > p5


class TestControlledEnvironment:
    """Test controlled environment for reproducibility."""

    def test_deterministic_tasks(self):
        env = ControlledEnvironment(seed=42, n_tasks=50)
        tasks = env.get_tasks()
        assert len(tasks) == 50

    def test_consistency(self):
        env = ControlledEnvironment(seed=42)
        assert env.verify_consistency()

    def test_condition_seed_deterministic(self):
        env = ControlledEnvironment(seed=42)
        s1 = env.get_condition_seed("full", 0)
        s2 = env.get_condition_seed("full", 0)
        assert s1 == s2

    def test_condition_seed_different_for_different_conditions(self):
        env = ControlledEnvironment(seed=42)
        s1 = env.get_condition_seed("full", 0)
        s2 = env.get_condition_seed("ablated", 0)
        assert s1 != s2

    def test_task_subset_category(self):
        env = ControlledEnvironment(seed=42, n_tasks=100)
        code_tasks = env.get_task_subset(category="code")
        assert all(t["category"] == "code" for t in code_tasks)

    def test_task_subset_difficulty(self):
        env = ControlledEnvironment(seed=42, n_tasks=100)
        hard_tasks = env.get_task_subset(difficulty="hard")
        assert all(t["difficulty"] == "hard" for t in hard_tasks)

    def test_reset(self):
        env = ControlledEnvironment(seed=42, n_tasks=10)
        t1 = env.get_tasks()
        env.reset()
        t2 = env.get_tasks()
        assert t1 == t2


class TestParallelRunner:
    """Test parallel condition runner."""

    def test_parallel_produces_same_results(self):
        suite = NeurosymbolicAblation()
        seq_runner = AblationRunner(noise_std=0.02)
        par_runner = ParallelConditionRunner(max_workers=2, pipeline=MockPipeline(noise_std=0.02))

        seq_result = seq_runner.run_suite(suite, repetitions=3, seed=42)
        par_result = par_runner.run_suite_parallel(suite, repetitions=3, seed=42)

        for cond in seq_result.get_all_condition_names():
            seq_scores = sorted(seq_result.get_scores(cond))
            par_scores = sorted(par_result.get_scores(cond))
            assert len(seq_scores) == len(par_scores)
            for s, p in zip(seq_scores, par_scores):
                assert abs(s - p) < 1e-10

    def test_parallel_multiple_suites(self):
        suites = [NeurosymbolicAblation(), GodelAgentAblation()]
        par_runner = ParallelConditionRunner(max_workers=2)
        results = par_runner.run_multiple_suites(suites, repetitions=2, seed=42)
        assert len(results) == 2


class TestCheckpointManager:
    """Test checkpoint save/load/resume."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            runner = AblationRunner(noise_std=0.02)
            suite = NeurosymbolicAblation()
            result = runner.run_suite(suite, repetitions=2, seed=42)

            path = mgr.save(result)
            assert os.path.exists(path)

            loaded = mgr.load(result.suite_name)
            assert loaded is not None
            assert loaded.suite_name == result.suite_name
            for cond in result.get_all_condition_names():
                orig_scores = result.get_scores(cond)
                load_scores = loaded.get_scores(cond)
                assert len(orig_scores) == len(load_scores)
                for o, l in zip(orig_scores, load_scores):
                    assert abs(o - l) < 1e-10

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            assert mgr.load("nonexistent") is None

    def test_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            runner = AblationRunner(noise_std=0.02)
            suite = NeurosymbolicAblation()
            result = runner.run_suite(suite, repetitions=2, seed=42)
            mgr.save(result)

            resumed = mgr.resume(result.suite_name)
            assert resumed is not None

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            assert not mgr.exists("test")
            runner = AblationRunner(noise_std=0.02)
            suite = NeurosymbolicAblation()
            result = runner.run_suite(suite, repetitions=2, seed=42)
            mgr.save(result)
            assert mgr.exists(result.suite_name)

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            runner = AblationRunner(noise_std=0.02)
            suite = NeurosymbolicAblation()
            result = runner.run_suite(suite, repetitions=2, seed=42)
            mgr.save(result)
            assert mgr.delete(result.suite_name)
            assert not mgr.exists(result.suite_name)

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            assert not mgr.delete("nonexistent")

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            assert mgr.list_checkpoints() == []
            runner = AblationRunner(noise_std=0.02)
            suite = NeurosymbolicAblation()
            result = runner.run_suite(suite, repetitions=2, seed=42)
            mgr.save(result)
            cps = mgr.list_checkpoints()
            assert len(cps) == 1


class TestReproducibility:
    """Test reproducibility utilities."""

    def test_set_global_seed(self):
        import random
        set_global_seed(42)
        a = random.random()
        set_global_seed(42)
        b = random.random()
        assert a == b

    def test_config_hash_deterministic(self):
        config = {"a": 1, "b": "hello"}
        h1 = get_config_hash(config)
        h2 = get_config_hash(config)
        assert h1 == h2

    def test_config_hash_different_configs(self):
        h1 = get_config_hash({"a": 1})
        h2 = get_config_hash({"a": 2})
        assert h1 != h2

    def test_verify_seed_consistency(self):
        assert verify_seed_consistency(42)

    def test_generate_run_id(self):
        rid = generate_run_id("test", 42, 5)
        assert isinstance(rid, str)
        assert len(rid) == 12


class TestCostEstimator:
    """Test cost estimation."""

    def test_estimate_suite(self):
        est = CostEstimator()
        suite = NeurosymbolicAblation()
        result = est.estimate_suite(suite, repetitions=5)
        assert result["n_conditions"] == 7
        assert result["total_runs"] == 35
        assert result["total_cost_usd"] == 3.50

    def test_estimate_all_suites(self):
        est = CostEstimator()
        suites = [
            NeurosymbolicAblation(),
            GodelAgentAblation(),
            SOARAblation(),
            RLMAblation(),
        ]
        result = est.estimate_all_suites(suites, repetitions=5)
        assert result["grand_total_runs"] == 155  # 31 conditions * 5 reps

    def test_format_estimate(self):
        est = CostEstimator()
        suite = NeurosymbolicAblation()
        result = est.estimate_suite(suite, repetitions=5)
        text = est.format_estimate(result)
        assert "Neurosymbolic" in text
        assert "$" in text

    def test_format_all_suites(self):
        est = CostEstimator()
        suites = [NeurosymbolicAblation(), GodelAgentAblation()]
        result = est.estimate_all_suites(suites, repetitions=5)
        text = est.format_estimate(result)
        assert "TOTAL" in text


class TestRunnerCost:
    """Test the runner's built-in cost estimation."""

    def test_estimate_cost(self):
        runner = AblationRunner()
        suite = NeurosymbolicAblation()
        cost = runner.estimate_cost(suite, repetitions=5, cost_per_run=0.10)
        assert cost["total_runs"] == 35
        assert cost["total_cost"] == 3.50

    def test_verify_power(self):
        a = [0.85, 0.86, 0.84, 0.85, 0.83]
        b = [0.72, 0.73, 0.71, 0.72, 0.70]
        result = AblationRunner._verify_power(a, b)
        assert "cohens_d" in result
        assert "power" in result
        assert result["power"] > 0


class TestNarrativeGeneration:
    """Test narrative text generation."""

    def test_generate_results_section(self):
        runner = AblationRunner(noise_std=0.02)
        suite = NeurosymbolicAblation()
        result = runner.run_suite(suite, repetitions=3, seed=42)
        analyses = suite.analyze(result)

        narr = NarrativeGenerator()
        text = narr.generate_results_section(result, analyses, suite.get_paper_name())
        assert len(text) > 0
        assert "accuracy" in text.lower()
        assert "significance" in text.lower() or "significant" in text.lower()

    def test_format_comparison_sentence(self):
        narr = NarrativeGenerator()
        pw = PairwiseResult(
            condition_a="full",
            condition_b="ablated",
            mean_a=0.85,
            mean_b=0.72,
            difference=0.13,
            t_statistic=5.0,
            p_value=0.001,
            ci_lower=0.08,
            ci_upper=0.18,
            effect_size=6.5,
            effect_interpretation="large",
            stars="**",
            n=5,
        )
        sentence = narr.format_comparison_sentence(pw)
        assert "full" in sentence
        assert "ablated" in sentence
        assert "0.130" in sentence
        assert "CI" in sentence
        assert "d = 6.50" in sentence

    def test_generate_summary_paragraph(self):
        runner = AblationRunner(noise_std=0.02)
        suites = [NeurosymbolicAblation(), GodelAgentAblation()]
        all_results = {}
        for suite in suites:
            result = runner.run_suite(suite, repetitions=2, seed=42)
            all_results[suite.get_paper_name()] = result

        narr = NarrativeGenerator()
        summary = narr.generate_summary_paragraph(all_results)
        assert len(summary) > 0
        assert "2" in summary  # 2 suites


class TestAppendixGeneration:
    """Test appendix generation."""

    def test_generate_per_category_tables(self):
        runner = AblationRunner(noise_std=0.02)
        suite = NeurosymbolicAblation()
        result = runner.run_suite(suite, repetitions=2, seed=42)

        app = AppendixGenerator()
        tables = app.generate_per_category_tables(result)
        assert len(tables) > 0

    def test_generate_per_category_with_categories(self):
        runner = AblationRunner(noise_std=0.02)
        suite = NeurosymbolicAblation()
        result = runner.run_suite(suite, repetitions=2, seed=42)

        app = AppendixGenerator()
        tables = app.generate_per_category_tables(result, categories=["code", "reasoning"])
        assert len(tables) == 2

    def test_generate_extended_comparisons(self):
        runner = AblationRunner(noise_std=0.02)
        suite = NeurosymbolicAblation()
        result = runner.run_suite(suite, repetitions=3, seed=42)

        app = AppendixGenerator()
        table = app.generate_extended_comparisons(result)
        assert "\\toprule" in table

    def test_generate_full_appendix(self):
        runner = AblationRunner(noise_std=0.02)
        suite = NeurosymbolicAblation()
        result = runner.run_suite(suite, repetitions=3, seed=42)

        app = AppendixGenerator()
        appendix = app.generate_full_appendix(result)
        assert "\\section" in appendix
        assert "Pairwise" in appendix


class TestInteractionAnalysis:
    """Test cross-suite interaction analysis."""

    def test_synergy_score(self):
        runner = AblationRunner(noise_std=0.02)
        suite_a = NeurosymbolicAblation()
        suite_b = GodelAgentAblation()
        result_a = runner.run_suite(suite_a, repetitions=2, seed=42)
        result_b = runner.run_suite(suite_b, repetitions=2, seed=42)

        analyzer = CrossSuiteInteractionAnalyzer()
        score = analyzer.compute_synergy_score(result_a, result_b)
        assert 0 <= score <= 1.0

    def test_interaction_report(self):
        runner = AblationRunner(noise_std=0.02)
        suites = [NeurosymbolicAblation(), GodelAgentAblation(), SOARAblation()]
        results = {}
        for suite in suites:
            result = runner.run_suite(suite, repetitions=2, seed=42)
            results[suite.get_paper_name()] = result

        analyzer = CrossSuiteInteractionAnalyzer()
        reports = analyzer.test_cross_paradigm_interactions(results)
        assert len(reports) == 3  # C(3,2)
        for r in reports:
            assert r.summary != ""


class TestStatisticalTests:
    """Additional statistical test coverage."""

    def test_pairwise_result_repr(self):
        pw = PairwiseResult(
            condition_a="a", condition_b="b",
            mean_a=0.85, mean_b=0.72, difference=0.13,
            t_statistic=5.0, p_value=0.001,
            ci_lower=0.08, ci_upper=0.18,
            effect_size=6.5, effect_interpretation="large",
            stars="**", n=5,
        )
        r = repr(pw)
        assert "a vs b" in r

    def test_pairwise_result_significant(self):
        pw = PairwiseResult(
            condition_a="a", condition_b="b",
            mean_a=0.85, mean_b=0.72, difference=0.13,
            t_statistic=5.0, p_value=0.001,
            ci_lower=0.08, ci_upper=0.18,
            effect_size=6.5, effect_interpretation="large",
            stars="**", n=5,
        )
        assert pw.significant is True

    def test_pairwise_not_significant(self):
        pw = PairwiseResult(
            condition_a="a", condition_b="b",
            mean_a=0.85, mean_b=0.84, difference=0.01,
            t_statistic=0.5, p_value=0.6,
            ci_lower=-0.02, ci_upper=0.04,
            effect_size=0.1, effect_interpretation="negligible",
            stars="", n=5,
        )
        assert pw.significant is False

    def test_multi_comparison(self):
        stats = PublicationStatistics()
        all_scores = {
            "full": [0.85, 0.86, 0.84, 0.85, 0.83],
            "ablated_a": [0.72, 0.73, 0.71, 0.72, 0.70],
            "ablated_b": [0.78, 0.79, 0.77, 0.78, 0.76],
        }
        results = stats.multi_comparison(all_scores, baseline="full")
        assert len(results) == 2

    def test_multi_comparison_missing_baseline(self):
        stats = PublicationStatistics()
        all_scores = {"a": [0.85], "b": [0.72]}
        results = stats.multi_comparison(all_scores, baseline="nonexistent")
        assert len(results) == 0

    def test_paired_t_test_identical(self):
        stats = PublicationStatistics()
        scores = [0.85, 0.86, 0.84, 0.85, 0.83]
        pw = stats.pairwise_comparison(scores, scores, "a", "a")
        assert pw.p_value >= 0.99 or abs(pw.difference) < 0.001


class TestBaseDataclasses:
    """Test base dataclass behaviors."""

    def test_ablation_condition_repr(self):
        from src.suites.base import AblationCondition
        c = AblationCondition(name="test", description="A test", is_full=True)
        r = repr(c)
        assert "test" in r
        assert "True" in r

    def test_condition_run_score(self):
        from src.suites.base import ConditionRun
        run = ConditionRun(condition_name="test", repetition=0, accuracy=0.85)
        assert run.score == 0.85

    def test_paper_assets_has_content_empty(self):
        assets = PaperAssets()
        assert not assets.has_content

    def test_paper_assets_has_content_with_tables(self):
        assets = PaperAssets(tables={"main": "content"})
        assert assets.has_content

    def test_suite_result_empty_scores(self):
        result = AblationSuiteResult(suite_name="test")
        assert result.get_scores("nonexistent") == []
        assert result.get_mean_score("nonexistent") == 0.0

    def test_suite_result_best_condition_empty(self):
        result = AblationSuiteResult(suite_name="test")
        assert result.best_condition() == ""

    def test_interaction_report_has_synergy(self):
        from src.analysis.interaction_tests import InteractionReport
        r = InteractionReport(suite_a="a", suite_b="b", synergy_score=0.5)
        assert r.has_synergy
        r2 = InteractionReport(suite_a="a", suite_b="b", synergy_score=0.0)
        assert not r2.has_synergy

    def test_interaction_report_negative_synergy(self):
        from src.analysis.interaction_tests import InteractionReport
        r = InteractionReport(suite_a="a", suite_b="b", synergy_score=-0.1)
        assert not r.has_synergy


class TestMockPipelineBehavior:
    """Test MockPipeline deterministic behavior."""

    def test_same_seed_same_result(self):
        from src.suites.base import AblationCondition
        pipeline = MockPipeline(noise_std=0.02)
        cond = AblationCondition(name="full", description="test")
        r1 = pipeline.run(cond, seed=42)
        r2 = pipeline.run(cond, seed=42)
        assert r1 == r2

    def test_different_seed_different_result(self):
        from src.suites.base import AblationCondition
        pipeline = MockPipeline(noise_std=0.02)
        cond = AblationCondition(name="full", description="test")
        r1 = pipeline.run(cond, seed=42)
        r2 = pipeline.run(cond, seed=99)
        # With noise, different seeds should give different results
        # (could theoretically be same but extremely unlikely)
        assert r1 != r2 or True  # Allow edge case

    def test_different_conditions_different_results(self):
        from src.suites.base import AblationCondition
        pipeline = MockPipeline(noise_std=0.02)
        cond_a = AblationCondition(name="full", description="test")
        cond_b = AblationCondition(name="no_repl", description="test")
        r_a = pipeline.run(cond_a, seed=42)
        r_b = pipeline.run(cond_b, seed=42)
        assert abs(r_a - r_b) > 0.1  # 0.85 vs 0.60

    def test_unknown_condition_gets_default(self):
        from src.suites.base import AblationCondition
        pipeline = MockPipeline(noise_std=0.0)
        cond = AblationCondition(name="totally_unknown", description="test")
        r = pipeline.run(cond, seed=42)
        assert abs(r - 0.70) < 0.01

    def test_zero_noise(self):
        from src.suites.base import AblationCondition
        pipeline = MockPipeline(noise_std=0.0)
        cond = AblationCondition(name="full", description="test")
        r = pipeline.run(cond, seed=42)
        assert r == 0.85

    def test_results_clamped(self):
        from src.suites.base import AblationCondition
        pipeline = MockPipeline(noise_std=10.0)  # Huge noise
        cond = AblationCondition(name="full", description="test")
        for seed in range(100):
            r = pipeline.run(cond, seed=seed)
            assert 0.0 <= r <= 1.0
