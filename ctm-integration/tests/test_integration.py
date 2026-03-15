"""End-to-end integration tests: build CTM, synthesize rules, augment prompts."""

import os

import pytest

from src.bdm.ctm_table import CTMTable
from src.bdm.scorer import BDMScorer
from src.bdm.calibration import BDMCalibrator
from src.synthesis.candidate_generator import CandidateGenerator, IOExample
from src.synthesis.synthesis_loop import SymbolicSynthesisLoop
from src.library.rule import VerifiedRule
from src.library.store import RuleStore
from src.library.index import RuleIndex
from src.library.evolution import LibraryEvolver
from src.integration.augmented_prompt import AugmentedPromptBuilder
from src.integration.comparison import AugmentationComparison
from src.integration.feedback_loop import FeedbackLoop
from src.integration.rule_invoker import RuleInvoker
from src.analysis.report import generate_report


class TestEndToEndPipeline:
    """Full end-to-end integration tests."""

    def test_build_ctm_and_score(self):
        """Build CTM table and use it for scoring."""
        # Build a small CTM table
        table = CTMTable()
        table.build(max_states=1, max_symbols=2, max_steps=20, block_size=8)
        assert table.is_built

        # Use it for scoring
        scorer = BDMScorer(ctm_table=table, block_size=4)
        score = scorer.score("01010101")
        assert score.bdm_value > 0

        # Verify ordering
        rep_score = scorer.score("00000000")
        rand_score = scorer.score("01101001")
        assert rep_score.bdm_value <= rand_score.bdm_value

    def test_synthesize_and_store(self, tmp_path):
        """Synthesize rules and store them in the library."""
        examples = [
            IOExample(input=1, output=2, domain="math"),
            IOExample(input=2, output=4, domain="math"),
            IOExample(input=3, output=6, domain="math"),
            IOExample(input=5, output=10, domain="math"),
        ]

        # Synthesize
        loop = SymbolicSynthesisLoop()
        result = loop.run(examples, max_iterations=2, candidates_per_iteration=5)
        assert result.final_best_accuracy >= 0.8

        # Store best rules
        store_path = str(tmp_path / "integration_rules.json")
        store = RuleStore(path=store_path)

        for sr in result.best_rules:
            if sr.accuracy >= 0.8:
                store.add(
                    VerifiedRule(
                        rule_id=sr.rule.rule_id,
                        domain="math",
                        description=sr.rule.description,
                        source_code=sr.rule.source_code,
                        accuracy=sr.accuracy,
                        bdm_score=sr.bdm_complexity,
                        mdl_score=sr.mdl_score,
                        tags=["math", "synthesized"],
                    )
                )

        assert store.size > 0
        assert os.path.exists(store_path)

    def test_augmented_prompt_with_library(self, populated_store):
        """Build augmented prompts using the rule library."""
        builder = AugmentedPromptBuilder(store=populated_store)

        prompt = builder.build_prompt("Compute the double of a number")
        assert "## Relevant Verified Rules" in prompt
        assert "## Task" in prompt

        # Standard prompt for comparison
        standard = builder.build_standard_prompt("Compute the double of a number")
        assert len(prompt) > len(standard)

    def test_full_pipeline(self, tmp_path):
        """Full pipeline: CTM -> synthesis -> library -> augmented prompt."""
        # 1. Build CTM table
        table = CTMTable.with_fallback_only()
        scorer = BDMScorer(ctm_table=table, block_size=4)

        # 2. Run calibration
        calibrator = BDMCalibrator(scorer)
        cal_report = calibrator.run_calibration(length=16)
        assert cal_report.ordering_correct

        # 3. Synthesize rules for multiple problems
        problems = {
            "double": [
                IOExample(input=1, output=2, domain="math"),
                IOExample(input=2, output=4, domain="math"),
                IOExample(input=3, output=6, domain="math"),
                IOExample(input=5, output=10, domain="math"),
            ],
            "square": [
                IOExample(input=1, output=1, domain="math"),
                IOExample(input=2, output=4, domain="math"),
                IOExample(input=3, output=9, domain="math"),
                IOExample(input=4, output=16, domain="math"),
            ],
        }

        store_path = str(tmp_path / "pipeline_rules.json")
        store = RuleStore(path=store_path)
        evolver = LibraryEvolver(store=store, min_accuracy=0.5)
        feedback = FeedbackLoop(store=store)

        loop = SymbolicSynthesisLoop(scorer=scorer)

        for name, examples in problems.items():
            result = loop.run(examples, max_iterations=2, candidates_per_iteration=5)

            new_rules = []
            for sr in result.best_rules:
                if sr.accuracy >= 0.5:
                    new_rules.append(
                        VerifiedRule(
                            rule_id=f"{name}_{sr.rule.rule_id}",
                            domain="math",
                            description=f"Rule for {name}",
                            source_code=sr.rule.source_code,
                            accuracy=sr.accuracy,
                            bdm_score=sr.bdm_complexity,
                            mdl_score=sr.mdl_score,
                            tags=[name, "math"],
                        )
                    )
            evolver.evolve(new_rules)

            # Feed back to feedback loop
            if result.iterations:
                last_iter = result.iterations[-1]
                feedback.feed_synthesis_results(last_iter.scored_rules)

        # 4. Verify library is populated
        assert store.size > 0

        # 5. Build augmented prompt
        builder = AugmentedPromptBuilder(store=store)
        prompt = builder.build_prompt("Multiply a number by 3")
        assert len(prompt) > 0

        # 6. Test comparison
        comparison = AugmentationComparison(store=store)
        results = comparison.run_comparison(["Triple a number", "Cube a number"])
        analysis = comparison.analyze(results)
        assert analysis.total_tasks == 2

        # 7. Generate report
        metrics = evolver.measure_library_quality()
        report = generate_report(
            calibration_report=cal_report,
            library_metrics=metrics,
        )
        assert "CTM Integration Report" in report

    def test_rule_invocation(self, sample_rules):
        """Rules should be invokable."""
        invoker = RuleInvoker()

        double_rule = sample_rules[0]  # y = 2x
        result = invoker.invoke(double_rule, 5)
        assert result == 10

        sum_rule = sample_rules[3]  # sum(list)
        result = invoker.invoke(sum_rule, [1, 2, 3])
        assert result == 6

    def test_feedback_loop(self, double_examples):
        """Feedback loop should track patterns."""
        store = RuleStore()
        feedback = FeedbackLoop(store=store)
        loop = SymbolicSynthesisLoop()

        result = loop.run(double_examples, max_iterations=2, candidates_per_iteration=3)

        for i, ir in enumerate(result.iterations):
            feedback.feed_synthesis_results(ir.scored_rules, iteration=i)

        context = feedback.get_context_for_generation()
        assert "successful_variants" in context
        assert "iteration_count" in context
        assert context["iteration_count"] == len(result.iterations)

    def test_calibration_ordering(self):
        """BDM calibration should confirm proper ordering."""
        scorer = BDMScorer(block_size=4)
        calibrator = BDMCalibrator(scorer)
        report = calibrator.run_calibration(length=16)

        assert report.num_tests >= 8
        assert report.ordering_correct

        # Check individual categories
        low = [r for r in report.results if r.expected_ordering == "low"]
        high = [r for r in report.results if r.expected_ordering == "high"]

        avg_low = sum(r.bdm_score for r in low) / len(low)
        avg_high = sum(r.bdm_score for r in high) / len(high)
        assert avg_low < avg_high
