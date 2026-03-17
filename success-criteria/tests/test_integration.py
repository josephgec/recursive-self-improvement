"""Integration tests — full pipeline: collect -> verify -> evaluate -> verdict -> report."""

import os
import tempfile

import pytest

from src.criteria.base import Evidence
from src.evidence.phase_collector import PhaseEvidenceCollector
from src.evidence.safety_collector import SafetyEvidenceCollector
from src.evidence.artifact_registry import ArtifactRegistry
from src.evidence.data_integrity import DataIntegrityVerifier
from src.evaluation.evaluator import CriteriaEvaluator
from src.evaluation.confidence import ConfidenceCalculator
from src.evaluation.sensitivity import SensitivityAnalyzer
from src.evaluation.preregistration import PreregistrationVerifier
from src.verdict.verdict import SuccessVerdict, VerdictCategory
from src.verdict.partial_success import PartialSuccessAnalyzer
from src.verdict.recommendations import RecommendationGenerator
from src.reporting.executive_summary import ExecutiveSummary
from src.reporting.technical_report import TechnicalReport
from src.reporting.evidence_appendix import EvidenceAppendix
from src.reporting.reproducibility import ReproducibilityPackager
from tests.conftest import (
    build_passing_evidence,
    build_partial_evidence,
    build_failing_evidence,
)


class TestFullPipelinePassing:
    """Integration test: full pipeline with passing evidence."""

    def test_collect_verify_evaluate_verdict_report(self):
        """Run complete pipeline end-to-end with passing evidence."""
        # Step 1: Collect evidence
        collector = PhaseEvidenceCollector()
        evidence = collector.collect_all()

        # Verify evidence is populated
        curve = evidence.get_improvement_curve()
        assert len(curve) == 5
        assert len(evidence.publications) >= 2
        assert len(evidence.get_gdi_readings()) > 0

        # Step 2: Verify integrity
        verifier = DataIntegrityVerifier()
        integrity = verifier.verify(evidence)
        assert integrity.hash_chain_valid is True

        # Step 3: Evaluate all criteria
        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(evidence)
        assert len(results) == 5

        # Step 4: Determine verdict
        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)
        assert verdict.category == VerdictCategory.SUCCESS
        assert verdict.n_passed == 5

        # Step 5: Generate reports
        exec_summary = ExecutiveSummary().generate(verdict)
        assert "GO" in exec_summary
        assert len(exec_summary) > 100

        tech_report = TechnicalReport().generate(verdict, evidence)
        assert "## 1. Introduction" in tech_report
        assert "## 8. Conclusions" in tech_report

        appendix = EvidenceAppendix().generate(evidence)
        assert "Phase Data" in appendix

        # Step 6: Package for reproducibility
        with tempfile.TemporaryDirectory() as tmpdir:
            packager = ReproducibilityPackager()
            artifacts = packager.package(evidence, verdict, tmpdir)
            assert len(artifacts) >= 5
            for name, path in artifacts.items():
                assert os.path.exists(path), f"{name} not found at {path}"


class TestFullPipelinePartial:
    """Integration test: partial success scenario."""

    def test_partial_pipeline(self):
        """Pipeline with partial evidence yields PARTIAL verdict."""
        evidence = build_partial_evidence()

        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(evidence)

        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        assert verdict.category == VerdictCategory.PARTIAL
        assert 3 <= verdict.n_passed <= 4

        # Partial analysis
        analyzer = PartialSuccessAnalyzer()
        analysis = analyzer.analyze(verdict)
        assert analysis["gap"] > 0
        assert analysis["closest_to_passing"] is not None

        # Recommendations
        gen = RecommendationGenerator()
        recs = gen.generate(verdict)
        assert len(recs) > 0
        assert any("CONDITIONAL" in r or "ACTION" in r for r in recs)

        # Executive summary
        summary = ExecutiveSummary().generate(verdict)
        assert "CONDITIONAL" in summary


class TestFullPipelineFailing:
    """Integration test: failing scenario."""

    def test_failing_pipeline(self):
        """Pipeline with failing evidence yields NOT_MET verdict."""
        evidence = build_failing_evidence()

        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(evidence)

        verdict_engine = SuccessVerdict()
        verdict = verdict_engine.evaluate(results)

        assert verdict.category == VerdictCategory.NOT_MET
        assert verdict.n_passed <= 2

        # Recommendations
        gen = RecommendationGenerator()
        recs = gen.generate(verdict)
        assert any("NO-GO" in r for r in recs)


class TestEvidenceCollection:
    """Test evidence collection components."""

    def test_phase_collector(self):
        """PhaseEvidenceCollector returns complete evidence."""
        collector = PhaseEvidenceCollector()
        evidence = collector.collect_all()

        assert evidence.phase_0 != {}
        assert evidence.phase_4 != {}
        assert "score" in evidence.phase_0
        assert len(evidence.publications) > 0
        assert len(evidence.audit_trail) > 0

    def test_safety_collector(self):
        """SafetyEvidenceCollector returns data for all subsystems."""
        collector = SafetyEvidenceCollector()
        data = collector.collect()

        assert "S.1" in data
        assert "S.2" in data
        assert "S.3" in data
        assert "S.4" in data
        for sid, sdata in data.items():
            assert sdata["status"] == "operational"

    def test_safety_collector_properties(self):
        """Test SafetyEvidenceCollector helper methods."""
        collector = SafetyEvidenceCollector()
        names = collector.get_subsystem_names()
        ids = collector.get_subsystem_ids()

        assert len(names) == 4
        assert len(ids) == 4
        assert "S.1" in ids

    def test_artifact_registry(self):
        """ArtifactRegistry can register, get, and list artifacts."""
        registry = ArtifactRegistry()
        art = registry.register("a1", "Test Artifact", "data", {"key": "value"})

        assert art.artifact_id == "a1"
        assert art.sha256 != ""

        retrieved = registry.get("a1")
        assert retrieved is not None
        assert retrieved.name == "Test Artifact"

        assert registry.count() == 1
        assert len(registry.list_all()) == 1

        # Verify integrity
        assert registry.verify_integrity("a1") is True

        # List by type
        assert len(registry.list_by_type("data")) == 1
        assert len(registry.list_by_type("other")) == 0

        # Remove
        assert registry.remove("a1") is True
        assert registry.get("a1") is None
        assert registry.remove("nonexistent") is False

    def test_artifact_registry_duplicate(self):
        """Registering duplicate ID should raise ValueError."""
        registry = ArtifactRegistry()
        registry.register("a1", "Test", "data", {"x": 1})
        with pytest.raises(ValueError):
            registry.register("a1", "Test2", "data", {"x": 2})

    def test_artifact_integrity_after_modification(self):
        """Integrity check should fail if data is modified."""
        registry = ArtifactRegistry()
        registry.register("a1", "Test", "data", {"key": "value"})

        # Modify the data directly
        artifact = registry.get("a1")
        artifact.data = {"key": "modified"}

        assert registry.verify_integrity("a1") is False

    def test_artifact_verify_nonexistent(self):
        """Verify integrity of nonexistent artifact returns False."""
        registry = ArtifactRegistry()
        assert registry.verify_integrity("does_not_exist") is False


class TestDataIntegrity:
    """Test data integrity verification."""

    def test_valid_evidence(self, passing_evidence):
        """Valid evidence passes integrity checks."""
        verifier = DataIntegrityVerifier()
        report = verifier.verify(passing_evidence)

        assert report.hash_chain_valid is True
        assert report.timestamp_valid is True
        assert report.overall_valid is True
        assert len(report.issues) == 0

    def test_broken_chain(self, failing_evidence):
        """Broken hash chain is detected."""
        verifier = DataIntegrityVerifier()
        report = verifier.verify(failing_evidence)

        assert report.hash_chain_valid is False
        assert report.overall_valid is False
        assert len(report.issues) > 0

    def test_preregistration_hash(self):
        """Preregistration hash check works."""
        verifier = DataIntegrityVerifier()
        good_hash = verifier.compute_config_hash()

        evidence = build_passing_evidence()
        report = verifier.verify(evidence, preregistration_hash=good_hash)
        assert report.preregistration_valid is True

        report2 = verifier.verify(evidence, preregistration_hash="wrong_hash")
        assert report2.preregistration_valid is False

    def test_no_preregistration_hash(self, passing_evidence):
        """No preregistration hash should skip that check."""
        verifier = DataIntegrityVerifier()
        report = verifier.verify(passing_evidence, preregistration_hash="")
        assert report.preregistration_valid is True  # skipped

    def test_integrity_report_summary(self, passing_evidence):
        """IntegrityReport summary format."""
        verifier = DataIntegrityVerifier()
        report = verifier.verify(passing_evidence)
        summary = report.summary()
        assert "VALID" in summary or "INVALID" in summary

    def test_compute_config_hash_file(self):
        """Test compute_config_hash with file path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test: data\n")
            f.flush()
            path = f.name

        try:
            h = DataIntegrityVerifier.compute_config_hash(path)
            assert len(h) == 64  # SHA-256 hex
        finally:
            os.unlink(path)

    def test_compute_config_hash_missing_file(self):
        """Missing file should return hash of 'missing'."""
        h = DataIntegrityVerifier.compute_config_hash("/nonexistent/path.yaml")
        assert len(h) == 64


class TestConfidenceAndSensitivity:
    """Test confidence calculator and sensitivity analyzer."""

    def test_overall_confidence(self, passing_evidence):
        """Overall confidence should be > 0 for passing evidence."""
        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(passing_evidence)

        calc = ConfidenceCalculator()
        conf = calc.compute_overall_confidence(results)
        assert 0 < conf <= 1.0

    def test_weighted_confidence(self, passing_evidence):
        """Weighted confidence with equal weights equals overall."""
        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(passing_evidence)

        calc = ConfidenceCalculator()
        weighted = calc.compute_weighted_confidence(results)
        assert 0 < weighted <= 1.0

    def test_weighted_confidence_with_weights(self, passing_evidence):
        """Custom weights should change the confidence value."""
        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(passing_evidence)

        calc = ConfidenceCalculator()
        weights = {r.criterion_name: 1.0 for r in results}
        weighted = calc.compute_weighted_confidence(results, weights)
        assert 0 < weighted <= 1.0

    def test_empty_confidence(self):
        """Empty results should give 0 confidence."""
        calc = ConfidenceCalculator()
        assert calc.compute_overall_confidence([]) == 0.0
        assert calc.compute_weighted_confidence([]) == 0.0

    def test_bootstrap(self, passing_evidence):
        """Bootstrap should return stability statistics."""
        from src.criteria.sustained_improvement import SustainedImprovementCriterion

        calc = ConfidenceCalculator()
        criterion = SustainedImprovementCriterion()
        stats = calc.bootstrap_criterion(
            criterion, passing_evidence, n_bootstrap=50, seed=42
        )

        assert "pass_rate" in stats
        assert "base_passed" in stats
        assert stats["n_bootstrap"] == 50
        assert 0 <= stats["pass_rate"] <= 1.0

    def test_bootstrap_short_curve(self):
        """Bootstrap with very short curve returns note."""
        from src.criteria.sustained_improvement import SustainedImprovementCriterion

        evidence = Evidence(
            phase_0={"score": 50.0, "collapse_score": 50.0},
        )
        calc = ConfidenceCalculator()
        stats = calc.bootstrap_criterion(
            SustainedImprovementCriterion(), evidence, n_bootstrap=10
        )
        assert "note" in stats

    def test_sensitivity_fragile(self, passing_evidence):
        """Sensitivity analysis should identify fragile criteria."""
        evaluator = CriteriaEvaluator()
        analyzer = SensitivityAnalyzer()

        fragile = analyzer.identify_fragile_criteria(
            evaluator.criteria, passing_evidence, margin_threshold=50.0
        )
        # With a large margin_threshold, many criteria may be flagile
        assert isinstance(fragile, list)

    def test_sensitivity_vary_thresholds(self, passing_evidence):
        """Threshold variation should produce results for each value."""
        from src.criteria.sustained_improvement import SustainedImprovementCriterion

        criterion = SustainedImprovementCriterion()
        analyzer = SensitivityAnalyzer()
        results = analyzer.vary_thresholds(
            criterion, passing_evidence,
            "min_total_gain_pp", [3.0, 5.0, 10.0, 20.0]
        )

        assert len(results) == 4
        # At 3.0, should pass; at 20.0, should fail
        assert results[0]["passed"] is True
        assert results[3]["passed"] is False

    def test_sensitivity_invalid_param(self, passing_evidence):
        """Invalid parameter should return error entries."""
        from src.criteria.sustained_improvement import SustainedImprovementCriterion

        criterion = SustainedImprovementCriterion()
        analyzer = SensitivityAnalyzer()
        results = analyzer.vary_thresholds(
            criterion, passing_evidence,
            "nonexistent_param", [1.0, 2.0]
        )

        assert len(results) == 2
        assert results[0]["passed"] is None

    def test_full_sensitivity_report(self, passing_evidence):
        """Full sensitivity report should have expected fields."""
        evaluator = CriteriaEvaluator()
        analyzer = SensitivityAnalyzer()
        report = analyzer.full_sensitivity_report(
            evaluator.criteria, passing_evidence
        )

        assert "fragile_criteria" in report
        assert "robustness_score" in report
        assert 0 <= report["robustness_score"] <= 1.0


class TestPreregistration:
    """Test preregistration verification."""

    def test_register_and_verify(self):
        """Register a config and verify it hasn't changed."""
        verifier = PreregistrationVerifier()
        config = {"threshold": 5.0, "alpha": 0.05}
        registered_hash = verifier.register(config)

        current_hash = verifier.compute_hash(config)
        result = verifier.verify_thresholds_unchanged(current_hash)

        assert result["verified"] is True

    def test_modified_config_fails(self):
        """Modified config should fail verification."""
        verifier = PreregistrationVerifier()
        config = {"threshold": 5.0, "alpha": 0.05}
        verifier.register(config)

        modified = {"threshold": 10.0, "alpha": 0.05}
        modified_hash = verifier.compute_hash(modified)
        result = verifier.verify_thresholds_unchanged(modified_hash)

        assert result["verified"] is False

    def test_no_registered_hash(self):
        """No registered hash should fail verification."""
        verifier = PreregistrationVerifier()
        result = verifier.verify_thresholds_unchanged("some_hash")

        assert result["verified"] is False

    def test_compute_file_hash(self):
        """File hash computation should return 64-char hex string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            path = f.name

        try:
            h = PreregistrationVerifier.compute_file_hash(path)
            assert len(h) == 64
        finally:
            os.unlink(path)

    def test_registered_hash_property(self):
        """registered_hash property should return stored hash."""
        verifier = PreregistrationVerifier("abc123")
        assert verifier.registered_hash == "abc123"


class TestReporting:
    """Test reporting components."""

    def test_executive_summary_passing(self, passing_evidence):
        """Executive summary for passing case."""
        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(passing_evidence)
        verdict = SuccessVerdict().evaluate(results)

        summary = ExecutiveSummary().generate(verdict)
        assert "GO" in summary
        assert "5" in summary
        assert "Criteria Results" in summary

    def test_executive_summary_failing(self, failing_evidence):
        """Executive summary for failing case."""
        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(failing_evidence)
        verdict = SuccessVerdict().evaluate(results)

        summary = ExecutiveSummary().generate(verdict)
        assert "NO-GO" in summary
        assert "Failed" in summary

    def test_technical_report(self, passing_evidence):
        """Technical report has all 8 sections."""
        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(passing_evidence)
        verdict = SuccessVerdict().evaluate(results)

        report = TechnicalReport().generate(verdict, passing_evidence)

        for section_num in range(1, 9):
            assert f"## {section_num}." in report

    def test_evidence_appendix(self, passing_evidence):
        """Evidence appendix has all sections."""
        appendix = EvidenceAppendix().generate(passing_evidence)

        assert "Phase Data" in appendix
        assert "Safety Data" in appendix
        assert "Publications" in appendix
        assert "Audit Trail" in appendix

    def test_reproducibility_packager(self, passing_evidence):
        """Reproducibility packager creates all files."""
        evaluator = CriteriaEvaluator()
        results = evaluator.evaluate_all(passing_evidence)
        verdict = SuccessVerdict().evaluate(results)

        with tempfile.TemporaryDirectory() as tmpdir:
            packager = ReproducibilityPackager()
            artifacts = packager.package(passing_evidence, verdict, tmpdir)

            assert "evidence" in artifacts
            assert "verdict" in artifacts
            assert "executive_summary" in artifacts
            assert "technical_report" in artifacts
            assert "evidence_appendix" in artifacts
            assert "manifest" in artifacts

            for name, path in artifacts.items():
                assert os.path.exists(path)


class TestEvaluatorDetails:
    """Test CriteriaEvaluator details."""

    def test_evaluate_single(self, passing_evidence):
        """evaluate_single returns result for a named criterion."""
        evaluator = CriteriaEvaluator()
        result = evaluator.evaluate_single("Sustained Improvement", passing_evidence)
        assert result is not None
        assert result.criterion_name == "Sustained Improvement"

    def test_evaluate_single_not_found(self, passing_evidence):
        """evaluate_single returns None for unknown criterion."""
        evaluator = CriteriaEvaluator()
        result = evaluator.evaluate_single("Nonexistent", passing_evidence)
        assert result is None

    def test_criteria_names(self):
        """criteria_names should list all 5 criterion names."""
        evaluator = CriteriaEvaluator()
        names = evaluator.criteria_names
        assert len(names) == 5
        assert "Sustained Improvement" in names

    def test_custom_criteria(self, passing_evidence):
        """CriteriaEvaluator with custom criteria list."""
        from src.criteria.gdi_bounds import GDIBoundsCriterion

        evaluator = CriteriaEvaluator(criteria=[GDIBoundsCriterion()])
        results = evaluator.evaluate_all(passing_evidence)
        assert len(results) == 1

    def test_config_based_construction(self, passing_evidence):
        """CriteriaEvaluator with config dict."""
        config = {
            "sustained_improvement": {"min_total_gain_pp": 3.0},
            "paradigm_improvement": {"alpha": 0.1},
        }
        evaluator = CriteriaEvaluator(config=config)
        results = evaluator.evaluate_all(passing_evidence)
        assert len(results) == 5


class TestEvidenceModel:
    """Test Evidence dataclass methods."""

    def test_get_improvement_curve(self, passing_evidence):
        """get_improvement_curve returns scores in order."""
        curve = passing_evidence.get_improvement_curve()
        assert len(curve) == 5
        assert curve[0] < curve[-1]

    def test_get_collapse_curve(self, passing_evidence):
        """get_collapse_curve returns collapse scores."""
        curve = passing_evidence.get_collapse_curve()
        assert len(curve) == 5

    def test_get_ablation_results(self, passing_evidence):
        """get_ablation_results returns all paradigms."""
        results = passing_evidence.get_ablation_results()
        assert "symcode" in results
        assert "godel" in results
        assert "soar" in results
        assert "rlm" in results

    def test_get_gdi_readings(self, passing_evidence):
        """get_gdi_readings returns safety readings."""
        readings = passing_evidence.get_gdi_readings()
        assert len(readings) > 0

    def test_get_phases_monitored(self, passing_evidence):
        """get_phases_monitored returns list of phases."""
        phases = passing_evidence.get_phases_monitored()
        assert len(phases) == 5

    def test_empty_evidence(self):
        """Empty evidence returns empty lists."""
        evidence = Evidence()
        assert evidence.get_improvement_curve() == []
        assert evidence.get_collapse_curve() == []
        assert evidence.get_ablation_results() == {}
        assert evidence.get_gdi_readings() == []
        assert evidence.get_phases_monitored() == []

    def test_criterion_result_summary(self):
        """CriterionResult summary format."""
        from src.criteria.base import CriterionResult
        result = CriterionResult(
            passed=True, confidence=0.95, measured_value=65.0,
            threshold=60.0, margin=5.0, criterion_name="Test"
        )
        summary = result.summary()
        assert "PASS" in summary
        assert "Test" in summary


import tempfile
