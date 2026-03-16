"""Integration tests: full 3-way comparison, winner table, RSI assessment, report."""

from __future__ import annotations

import pytest

from src.hybrid.pipeline import HybridPipeline
from src.integrative.pipeline import IntegrativePipeline
from src.evaluation.benchmark_suite import BenchmarkSuite, BenchmarkResults
from src.evaluation.statistical_tests import StatisticalComparator, TestResult
from src.analysis.head_to_head import HeadToHeadAnalyzer, HeadToHeadReport
from src.analysis.failure_modes import FailureModeAnalyzer
from src.analysis.cost_analysis import CostAnalyzer
from src.analysis.rsi_suitability import RSISuitabilityAssessor, RSIAssessment
from src.analysis.report import generate_report
from src.utils.task_domains import MultiDomainTaskLoader, Task
from src.deliverables.symcode_summary import package_symcode_results
from src.deliverables.bdm_summary import package_bdm_results
from src.deliverables.final_report import generate_phase1a_report
from tests.conftest import ProsePipeline


# ── Benchmark Suite tests ──

class TestBenchmarkSuite:
    def test_register_system(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        assert "hybrid" in suite.registered_systems

    def test_run_full_suite(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("integrative", IntegrativePipeline())
        suite.register_system("prose", ProsePipeline())

        results = suite.run_full_suite(domains=["arithmetic"])
        assert isinstance(results, BenchmarkResults)
        assert "hybrid" in results.generalization
        assert "integrative" in results.interpretability
        assert "prose" in results.robustness

    def test_run_single_axis_generalization(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("integrative", IntegrativePipeline())

        results = suite.run_single_axis("generalization", domains=["arithmetic"])
        assert "hybrid" in results
        assert "integrative" in results

    def test_run_single_axis_interpretability(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        results = suite.run_single_axis("interpretability", domains=["arithmetic"])
        assert "hybrid" in results

    def test_run_single_axis_robustness(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        results = suite.run_single_axis("robustness", domains=["arithmetic"])
        assert "hybrid" in results

    def test_run_single_axis_invalid(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        with pytest.raises(ValueError):
            suite.run_single_axis("invalid_axis")

    def test_get_summary(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])
        summary = results.get_summary()
        assert "hybrid" in summary
        assert "prose" in summary
        assert "in_domain_accuracy" in summary["hybrid"]


# ── Statistical Tests ──

class TestStatisticalComparator:
    def test_paired_test_equal(self):
        comp = StatisticalComparator()
        a = [True, False, True, True, False]
        b = [True, False, True, True, False]
        result = comp.paired_test(a, b)
        assert isinstance(result, TestResult)
        assert result.test_name == "McNemar"
        assert result.p_value >= 0

    def test_paired_test_different(self):
        comp = StatisticalComparator()
        a = [True] * 20 + [False] * 5
        b = [False] * 20 + [True] * 5
        result = comp.paired_test(a, b)
        assert result.p_value < 1.0

    def test_paired_test_unequal_length(self):
        comp = StatisticalComparator()
        with pytest.raises(AssertionError):
            comp.paired_test([True, False], [True])

    def test_multi_system_test(self):
        comp = StatisticalComparator()
        accs = {
            "hybrid": [True, False, True, True, False, True],
            "integrative": [True, True, False, True, False, True],
            "prose": [False, False, True, True, False, False],
        }
        result = comp.multi_system_test(accs)
        assert isinstance(result, TestResult)
        assert result.test_name == "Cochran_Q"

    def test_multi_system_test_single_system(self):
        comp = StatisticalComparator()
        result = comp.multi_system_test({"a": [True, False]})
        assert result.p_value == 1.0

    def test_effect_size(self):
        comp = StatisticalComparator()
        a = [True] * 8 + [False] * 2  # 80%
        b = [True] * 5 + [False] * 5  # 50%
        h = comp.effect_size(a, b)
        assert h > 0

    def test_effect_size_equal(self):
        comp = StatisticalComparator()
        a = [True, False, True, False]
        h = comp.effect_size(a, a)
        assert h == 0.0

    def test_confidence_intervals(self):
        comp = StatisticalComparator()
        correct = [True] * 7 + [False] * 3
        lower, upper = comp.confidence_intervals(correct, confidence=0.95)
        assert 0.0 <= lower <= upper <= 1.0
        assert lower < 0.7  # should include the true proportion
        assert upper > 0.7

    def test_confidence_intervals_empty(self):
        comp = StatisticalComparator()
        lower, upper = comp.confidence_intervals([])
        assert lower == 0.0
        assert upper == 0.0

    def test_confidence_intervals_all_true(self):
        comp = StatisticalComparator()
        lower, upper = comp.confidence_intervals([True] * 10)
        assert upper == 1.0

    def test_required_sample_size(self):
        comp = StatisticalComparator()
        n = comp.required_sample_size(effect_size=0.5, power=0.8, alpha=0.05)
        assert isinstance(n, int)
        assert n > 0

    def test_required_sample_size_small_effect(self):
        comp = StatisticalComparator()
        n_large = comp.required_sample_size(effect_size=0.1)
        n_small = comp.required_sample_size(effect_size=0.5)
        assert n_large > n_small

    def test_required_sample_size_zero_effect(self):
        comp = StatisticalComparator()
        n = comp.required_sample_size(effect_size=0.0)
        assert n == 1000  # fallback

    def test_chi2_p_value_zero(self):
        assert StatisticalComparator._chi2_p_value(0.0) == 1.0

    def test_chi2_p_value_large(self):
        p = StatisticalComparator._chi2_p_value(100.0, df=1)
        assert p < 0.001

    def test_chi2_p_value_df_zero(self):
        p = StatisticalComparator._chi2_p_value(5.0, df=0)
        assert p == 1.0


# ── Head-to-Head Analysis ──

class TestHeadToHeadAnalyzer:
    def _make_results(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("integrative", IntegrativePipeline())
        suite.register_system("prose", ProsePipeline())
        return suite.run_full_suite(domains=["arithmetic"])

    def test_analyze(self):
        results = self._make_results()
        analyzer = HeadToHeadAnalyzer()
        report = analyzer.analyze(results)
        assert isinstance(report, HeadToHeadReport)
        assert report.overall_winner != ""
        assert len(report.winner_table) == 3

    def test_winner_table_all_axes(self):
        results = self._make_results()
        analyzer = HeadToHeadAnalyzer()
        report = analyzer.analyze(results)
        assert "generalization" in report.winner_table
        assert "interpretability" in report.winner_table
        assert "robustness" in report.winner_table

    def test_generate_winner_table_markdown(self):
        results = self._make_results()
        analyzer = HeadToHeadAnalyzer()
        report = analyzer.analyze(results)
        table = analyzer.generate_winner_table(report)
        assert "| Axis |" in table
        assert "Overall Winner" in table

    def test_scores_populated(self):
        results = self._make_results()
        analyzer = HeadToHeadAnalyzer()
        report = analyzer.analyze(results)
        for system in report.scores:
            assert "generalization" in report.scores[system] or \
                   "interpretability" in report.scores[system]

    def test_plot_radar_chart(self):
        results = self._make_results()
        analyzer = HeadToHeadAnalyzer()
        report = analyzer.analyze(results)
        fig = analyzer.plot_radar_chart(report)
        # May be None if matplotlib is not available
        # Just test it doesn't crash

    def test_plot_per_axis(self):
        results = self._make_results()
        analyzer = HeadToHeadAnalyzer()
        report = analyzer.analyze(results)
        fig = analyzer.plot_per_axis_comparison(report)
        # May be None if matplotlib is not available


# ── Failure Mode Analysis ──

class TestFailureModeAnalyzer:
    def test_analyze(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("integrative", IntegrativePipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        analyzer = FailureModeAnalyzer()
        reports = analyzer.analyze(results)
        assert "hybrid" in reports
        assert "integrative" in reports
        assert "prose" in reports

    def test_failure_categories(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        analyzer = FailureModeAnalyzer()
        reports = analyzer.analyze(results)
        for system, report in reports.items():
            assert len(report.categories) > 0

    def test_complementary_strengths(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        analyzer = FailureModeAnalyzer()
        complements = analyzer.find_complementary_strengths(results)
        assert isinstance(complements, dict)

    def test_categorize_hybrid_failures_tool_error(self):
        analyzer = FailureModeAnalyzer()
        failures = [
            {"predicted": "error occurred", "domain": "arithmetic"},
            {"predicted": "unknown result", "domain": "arithmetic"},
            {"predicted": "wrong answer 42", "domain": "arithmetic"},
        ]
        cats = analyzer._categorize_hybrid_failures(failures)
        names = {c.name: c.count for c in cats}
        assert names["tool_error"] == 1
        assert names["no_tool_call"] == 1
        assert names["integration_error"] == 1

    def test_categorize_integrative_failures(self):
        analyzer = FailureModeAnalyzer()
        failures = [
            {"predicted": "unknown", "domain": "arithmetic"},
            {"predicted": "wrong 42", "domain": "arithmetic"},
        ]
        cats = analyzer._categorize_integrative_failures(failures)
        names = {c.name: c.count for c in cats}
        assert names["constraint_miss"] == 1
        assert names["general_error"] == 1

    def test_categorize_prose_failures(self):
        analyzer = FailureModeAnalyzer()
        failures = [
            {"predicted": "wrong", "domain": "arithmetic"},
            {"predicted": "wrong", "domain": "logic"},
            {"predicted": "wrong", "domain": "unknown_domain"},
        ]
        cats = analyzer._categorize_prose_failures(failures)
        names = {c.name: c.count for c in cats}
        assert names["computation_error"] == 1
        assert names["logic_error"] == 1
        assert names["other"] == 1


# ── Cost Analysis ──

class TestCostAnalyzer:
    def test_analyze(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("integrative", IntegrativePipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        analyzer = CostAnalyzer()
        report = analyzer.analyze(results)
        assert report.most_efficient != ""
        assert len(report.system_costs) == 3

    def test_hybrid_costs_higher_than_prose(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        analyzer = CostAnalyzer()
        report = analyzer.analyze(results)
        hybrid_cost = report.system_costs["hybrid"]["cost_per_task"]
        prose_cost = report.system_costs["prose"]["cost_per_task"]
        assert hybrid_cost > prose_cost

    def test_plot_tradeoff(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        analyzer = CostAnalyzer()
        report = analyzer.analyze(results)
        fig = analyzer.plot_accuracy_latency_tradeoff(report)


# ── RSI Suitability ──

class TestRSISuitability:
    def test_assess(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("integrative", IntegrativePipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        assessor = RSISuitabilityAssessor()
        assessments = assessor.assess(results)
        assert "hybrid" in assessments
        assert "integrative" in assessments
        assert "prose" in assessments

    def test_hybrid_high_modularity(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        assessor = RSISuitabilityAssessor()
        assessments = assessor.assess(results)
        assert assessments["hybrid"].modularity >= 0.7

    def test_prose_low_modularity(self):
        suite = BenchmarkSuite()
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        assessor = RSISuitabilityAssessor()
        assessments = assessor.assess(results)
        assert assessments["prose"].modularity < 0.5

    def test_assessment_has_recommendation(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        assessor = RSISuitabilityAssessor()
        assessments = assessor.assess(results)
        for system, assessment in assessments.items():
            assert isinstance(assessment, RSIAssessment)
            assert assessment.recommendation != ""
            assert 0.0 <= assessment.overall_score <= 1.0

    def test_strengths_and_weaknesses(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        assessor = RSISuitabilityAssessor()
        assessments = assessor.assess(results)
        # Hybrid should have strengths
        assert len(assessments["hybrid"].strengths) > 0
        # Prose should have weaknesses
        assert len(assessments["prose"].weaknesses) > 0

    def test_custom_weights(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        assessor = RSISuitabilityAssessor(weights={
            "modularity": 0.5,
            "verifiability": 0.1,
            "composability": 0.1,
            "contamination_resistance": 0.1,
            "transparency": 0.2,
        })
        assessments = assessor.assess(results)
        assert "hybrid" in assessments


# ── Report Generation ──

class TestReportGeneration:
    def test_generate_report(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("integrative", IntegrativePipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        report = generate_report(results)
        assert "# Architecture Comparison Report" in report
        assert "Generalization" in report
        assert "Interpretability" in report
        assert "Robustness" in report
        assert "RSI" in report

    def test_report_contains_all_systems(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        suite.register_system("prose", ProsePipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        report = generate_report(results)
        assert "hybrid" in report
        assert "prose" in report

    def test_report_custom_title(self):
        suite = BenchmarkSuite()
        suite.register_system("hybrid", HybridPipeline())
        results = suite.run_full_suite(domains=["arithmetic"])

        report = generate_report(results, title="Custom Title")
        assert "# Custom Title" in report


# ── Task Domains ──

class TestMultiDomainTaskLoader:
    def test_load_arithmetic(self, task_loader):
        tasks = task_loader.load_domain("arithmetic")
        assert len(tasks) >= 20
        for t in tasks:
            assert t.domain == "arithmetic"

    def test_load_algebra(self, task_loader):
        tasks = task_loader.load_domain("algebra")
        assert len(tasks) >= 20

    def test_load_logic(self, task_loader):
        tasks = task_loader.load_domain("logic")
        assert len(tasks) >= 20

    def test_load_probability(self, task_loader):
        tasks = task_loader.load_domain("probability")
        assert len(tasks) >= 20

    def test_load_unknown_domain(self, task_loader):
        with pytest.raises(ValueError):
            task_loader.load_domain("nonexistent")

    def test_load_cross_domain(self, task_loader):
        tasks = task_loader.load_cross_domain(["arithmetic", "logic"])
        domains = {t.domain for t in tasks}
        assert "arithmetic" in domains
        assert "logic" in domains

    def test_load_cross_domain_all(self, task_loader):
        tasks = task_loader.load_cross_domain()
        domains = {t.domain for t in tasks}
        assert len(domains) >= 4

    def test_load_paired_perturbations(self, task_loader):
        originals, perturbed = task_loader.load_paired_perturbations("arithmetic")
        assert len(originals) == len(perturbed)
        for o, p in zip(originals, perturbed):
            assert o.expected_answer == p.expected_answer
            assert p.problem != o.problem

    def test_available_domains(self, task_loader):
        domains = task_loader.available_domains
        assert "arithmetic" in domains
        assert "algebra" in domains
        assert "logic" in domains
        assert "probability" in domains

    def test_task_dataclass(self):
        t = Task(task_id="test", domain="arithmetic",
                 problem="What is 1 + 1?", expected_answer="2")
        assert t.task_id == "test"
        assert t.difficulty == "medium"  # default


# ── Deliverables ──

class TestDeliverables:
    def test_package_symcode(self, temp_dir):
        result = package_symcode_results(temp_dir)
        assert result["architecture"] == "hybrid"
        assert result["name"] == "SymCode"
        assert "metrics" in result
        assert "key_findings" in result

    def test_package_bdm(self, temp_dir):
        result = package_bdm_results(temp_dir)
        assert result["architecture"] == "integrative"
        assert result["name"] == "BDM (Bounded Deductive Model)"
        assert "metrics" in result

    def test_generate_phase1a_report(self, temp_dir):
        report = generate_phase1a_report(temp_dir)
        assert "Phase 1a" in report
        assert "SymCode" in report
        assert "BDM" in report
        assert "Recommendations" in report

    def test_symcode_summary_structure(self, temp_dir):
        result = package_symcode_results(temp_dir)
        assert "generalization" in result["metrics"]
        assert "interpretability" in result["metrics"]
        assert "robustness" in result["metrics"]

    def test_bdm_summary_structure(self, temp_dir):
        result = package_bdm_results(temp_dir)
        assert "generalization" in result["metrics"]
        assert "in_domain_accuracy" in result["metrics"]["generalization"]

    def test_package_symcode_with_file(self, temp_dir):
        import json, os
        data = {"metrics": {"accuracy": 0.9}, "key_findings": ["finding1"]}
        with open(os.path.join(temp_dir, "hybrid_results.json"), "w") as f:
            json.dump(data, f)
        result = package_symcode_results(temp_dir)
        assert result["metrics"]["accuracy"] == 0.9
        assert result["key_findings"] == ["finding1"]

    def test_package_bdm_with_file(self, temp_dir):
        import json, os
        data = {"metrics": {"accuracy": 0.8}, "key_findings": ["finding2"]}
        with open(os.path.join(temp_dir, "integrative_results.json"), "w") as f:
            json.dump(data, f)
        result = package_bdm_results(temp_dir)
        assert result["metrics"]["accuracy"] == 0.8
        assert result["key_findings"] == ["finding2"]


# ── IntegrativePipeline tests ──

class TestIntegrativePipeline:
    def test_solve_arithmetic(self):
        pipeline = IntegrativePipeline()
        result = pipeline.solve("What is 3 + 4?")
        assert result.answer != ""
        assert result.total_time >= 0

    def test_solve_returns_integrative_result(self):
        from src.integrative.pipeline import IntegrativeResult
        pipeline = IntegrativePipeline()
        result = pipeline.solve("Calculate 5 * 6")
        assert isinstance(result, IntegrativeResult)
        assert result.constrained_output is not None

    def test_solve_with_custom_model(self):
        def custom_model(prompt):
            return "Computing: 3 + 4 = 7. The answer is 7."

        pipeline = IntegrativePipeline(model=custom_model)
        result = pipeline.solve("What is 3 + 4?")
        assert "7" in result.answer

    def test_extract_answer_patterns(self):
        pipeline = IntegrativePipeline()
        assert pipeline._extract_answer("the answer is 42.") == "42"
        assert pipeline._extract_answer("ANSWER: 7") == "7"
        assert pipeline._extract_answer("just text\nlast line") == "last line"
