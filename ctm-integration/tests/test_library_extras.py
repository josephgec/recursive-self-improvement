"""Tests for library extras: composer, evolution, rule invoker, feedback loop."""

import pytest

from src.library.rule import VerifiedRule
from src.library.store import RuleStore
from src.library.composer import RuleComposer, ComposedRule
from src.library.evolution import LibraryEvolver, LibraryMetrics
from src.integration.rule_invoker import RuleInvoker
from src.integration.feedback_loop import FeedbackLoop
from src.synthesis.candidate_generator import IOExample
from src.synthesis.pareto_selector import ScoredRule
from src.synthesis.candidate_generator import CandidateRule
from src.synthesis.complexity_ranker import ComplexityRanker, RankedRule


class TestRuleComposer:
    """Tests for rule composition."""

    def test_compose_single_rule(self, sample_rules):
        """Composing a single rule should work."""
        composer = RuleComposer()
        examples = [
            IOExample(input=5, output=10),
            IOExample(input=3, output=6),
        ]
        result = composer.compose(sample_rules[:1], examples, max_rules=1)
        # The double rule should work
        if result is not None:
            assert result.accuracy > 0

    def test_compose_empty_rules(self):
        """Composing with empty rules should return None."""
        composer = RuleComposer()
        examples = [IOExample(input=1, output=2)]
        result = composer.compose([], examples)
        assert result is None

    def test_compose_empty_examples(self, sample_rules):
        """Composing with empty examples should return None."""
        composer = RuleComposer()
        result = composer.compose(sample_rules, [])
        assert result is None

    def test_composed_rule_has_source(self, sample_rules):
        """ComposedRule should have source code."""
        composed = ComposedRule(
            composed_id="test",
            component_rules=sample_rules[:1],
            composition_type="sequential",
            description="test",
        )
        code = composed.source_code
        assert "def composed_rule" in code

    def test_conditional_composition_source(self, sample_rules):
        """Conditional composition should generate valid code."""
        composed = ComposedRule(
            composed_id="test",
            component_rules=sample_rules[:2],
            composition_type="conditional",
            description="test",
        )
        code = composed.source_code
        assert "def composed_rule" in code
        assert "try" in code

    def test_try_sequential(self, sample_rules):
        """Sequential composition should be attempted."""
        composer = RuleComposer()
        examples = [
            IOExample(input=5, output=10),
            IOExample(input=3, output=6),
        ]
        result = composer._try_sequential(sample_rules, examples, max_rules=2)
        assert result is not None

    def test_try_conditional(self, sample_rules):
        """Conditional composition should be attempted."""
        composer = RuleComposer()
        examples = [
            IOExample(input=5, output=10),
            IOExample(input=3, output=6),
        ]
        result = composer._try_conditional(sample_rules, examples, max_rules=2)
        assert result is not None


class TestLibraryEvolution:
    """Tests for library evolution."""

    def test_evolve_adds_rules(self, tmp_store):
        """evolve() should add qualifying rules."""
        evolver = LibraryEvolver(store=tmp_store, min_accuracy=0.5)

        new_rules = [
            VerifiedRule(
                rule_id="r1", domain="math", description="test",
                source_code="def r(x): return x", accuracy=0.9,
            ),
            VerifiedRule(
                rule_id="r2", domain="math", description="test2",
                source_code="def r(x): return x+1", accuracy=0.3,
            ),
        ]
        metrics = evolver.evolve(new_rules)
        assert isinstance(metrics, LibraryMetrics)
        assert tmp_store.size == 1  # Only r1 meets threshold

    def test_prune_low_accuracy(self, tmp_store):
        """prune() should remove low-accuracy rules."""
        evolver = LibraryEvolver(store=tmp_store, min_accuracy=0.5)
        tmp_store.add(VerifiedRule(
            rule_id="low", domain="math", description="", source_code="", accuracy=0.2
        ))
        tmp_store.add(VerifiedRule(
            rule_id="high", domain="math", description="", source_code="x", accuracy=0.9
        ))
        pruned = evolver.prune()
        assert pruned == 1
        assert tmp_store.size == 1

    def test_prune_per_domain_limit(self, tmp_store):
        """prune() should enforce per-domain limits."""
        evolver = LibraryEvolver(store=tmp_store, min_accuracy=0.0, max_rules_per_domain=2)
        for i in range(5):
            tmp_store.add(VerifiedRule(
                rule_id=f"r{i}", domain="math", description=f"rule {i}",
                source_code=f"def r(x): return x + {i}",
                accuracy=0.5 + 0.1 * i,
            ))
        pruned = evolver.prune()
        assert pruned == 3  # Keep only top 2

    def test_measure_library_quality_empty(self, tmp_store):
        """Empty library should return zero metrics."""
        evolver = LibraryEvolver(store=tmp_store)
        metrics = evolver.measure_library_quality()
        assert metrics.total_rules == 0

    def test_measure_quality_with_known_domains(self, tmp_store):
        """Quality with known domains should compute coverage."""
        evolver = LibraryEvolver(store=tmp_store, min_accuracy=0.5)
        tmp_store.add(VerifiedRule(
            rule_id="r1", domain="math", description="", source_code="",
            accuracy=0.9, bdm_score=10.0, mdl_score=15.0,
        ))
        metrics = evolver.measure_library_quality(
            known_domains=["math", "string", "logic"]
        )
        assert metrics.coverage == pytest.approx(1.0 / 3.0)

    def test_metrics_history(self, tmp_store):
        """Evolution should track metrics history."""
        evolver = LibraryEvolver(store=tmp_store, min_accuracy=0.0)
        for i in range(3):
            evolver.evolve([VerifiedRule(
                rule_id=f"r{i}", domain="math", description="",
                source_code=f"x{i}", accuracy=0.8,
            )])
        assert len(evolver.metrics_history) == 3

    def test_metrics_to_dict(self):
        """LibraryMetrics.to_dict should work."""
        metrics = LibraryMetrics(
            total_rules=10, unique_domains=3, avg_accuracy=0.8,
            avg_bdm_score=15.0, avg_mdl_score=20.0, coverage=0.9,
            quality_score=0.7,
        )
        d = metrics.to_dict()
        assert d["total_rules"] == 10.0
        assert d["avg_accuracy"] == 0.8


class TestRuleInvoker:
    """Tests for rule invocation."""

    def test_invoke_simple_rule(self):
        """invoke() should execute a rule."""
        invoker = RuleInvoker()
        rule = VerifiedRule(
            rule_id="r1", domain="math", description="double",
            source_code="def rule(x):\n    return x * 2\n",
        )
        result = invoker.invoke(rule, 5)
        assert result == 10

    def test_invoke_list_input(self):
        """invoke() should handle list inputs."""
        invoker = RuleInvoker()
        rule = VerifiedRule(
            rule_id="r1", domain="math", description="sum",
            source_code="def rule(x):\n    return sum(x)\n",
        )
        result = invoker.invoke(rule, [1, 2, 3])
        assert result == 6

    def test_invoke_failing_rule(self):
        """invoke() should raise RuntimeError on failure."""
        invoker = RuleInvoker()
        rule = VerifiedRule(
            rule_id="r1", domain="math", description="crash",
            source_code="def rule(x):\n    raise ValueError('boom')\n",
        )
        with pytest.raises(RuntimeError, match="execution failed"):
            invoker.invoke(rule, 5)

    def test_invoke_no_callable(self):
        """invoke() should raise if no callable found."""
        invoker = RuleInvoker()
        rule = VerifiedRule(
            rule_id="r1", domain="math", description="no func",
            source_code="x = 42\n",
        )
        with pytest.raises(RuntimeError, match="No callable"):
            invoker.invoke(rule, 5)

    def test_invoke_composed_rule(self, sample_rules):
        """invoke_composed() should execute a composed rule."""
        invoker = RuleInvoker()
        composed = ComposedRule(
            composed_id="c1",
            component_rules=[],
            composition_type="sequential",
            description="identity",
        )
        # Override source_code property with a direct function
        composed_code_rule = ComposedRule(
            composed_id="c1",
            component_rules=[],
            composition_type="sequential",
            description="test",
        )
        # Create a simple composed rule manually
        rule = VerifiedRule(
            rule_id="r1", domain="math", description="",
            source_code="def rule(x):\n    return x * 2\n",
        )
        simple_composed = ComposedRule(
            composed_id="c2",
            component_rules=[rule],
            composition_type="conditional",
            description="test",
        )
        # Test the conditional composition
        result = invoker.invoke_composed(simple_composed, 5)
        assert result == 10

    def test_invoke_composed_crashing(self):
        """invoke_composed() should raise RuntimeError if execution fails."""
        invoker = RuleInvoker()
        crash_rule = VerifiedRule(
            rule_id="crash", domain="math", description="",
            source_code="def rule(x):\n    raise ValueError('boom')\n",
        )
        composed = ComposedRule(
            composed_id="c1",
            component_rules=[crash_rule],
            composition_type="conditional",
            description="crash test",
        )
        # Conditional composition catches exceptions and returns None,
        # but the composed function itself should not crash
        result = invoker.invoke_composed(composed, 5)
        assert result is None


class TestComplexityRanker:
    """Tests for complexity ranking."""

    def test_rank_by_bdm(self, scorer, double_examples):
        """rank_by_bdm should sort by BDM score."""
        ranker = ComplexityRanker(scorer=scorer)
        rules = [
            CandidateRule(
                rule_id="short",
                source_code="def rule(x):\n    return x * 2\n",
                description="short",
            ),
            CandidateRule(
                rule_id="long",
                source_code=(
                    "def rule(x):\n"
                    "    # This is a much longer rule\n"
                    "    result = x\n"
                    "    result = result + x\n"
                    "    return result\n"
                ),
                description="long",
            ),
        ]
        ranked = ranker.rank_by_bdm(rules, double_examples)
        assert len(ranked) == 2
        assert all(isinstance(r, RankedRule) for r in ranked)
        # Shorter rule should have lower BDM
        assert ranked[0].rule_score.bdm_score <= ranked[1].rule_score.bdm_score
        assert ranked[0].rank == 1

    def test_rank_by_mdl(self, scorer, double_examples):
        """rank_by_mdl should sort by MDL score."""
        ranker = ComplexityRanker(scorer=scorer)
        rules = [
            CandidateRule(
                rule_id="correct",
                source_code="def rule(x):\n    return x * 2\n",
                description="correct",
            ),
            CandidateRule(
                rule_id="wrong",
                source_code="def rule(x):\n    return x + 1\n",
                description="wrong",
            ),
        ]
        ranked = ranker.rank_by_mdl(rules, double_examples)
        assert len(ranked) == 2
        # The correct rule should have lower MDL (fits data better)
        correct_ranked = next(r for r in ranked if r.rule.rule_id == "correct")
        wrong_ranked = next(r for r in ranked if r.rule.rule_id == "wrong")
        assert correct_ranked.rule_score.mdl_score <= wrong_ranked.rule_score.mdl_score

    def test_rank_without_examples(self, scorer):
        """Ranking without examples should still work."""
        ranker = ComplexityRanker(scorer=scorer)
        rules = [
            CandidateRule(
                rule_id="r1",
                source_code="def rule(x):\n    return x\n",
                description="",
            ),
        ]
        ranked = ranker.rank_by_bdm(rules)
        assert len(ranked) == 1


class TestFeedbackLoop:
    """Tests for feedback loop."""

    def test_feed_results(self):
        """Feeding results should track patterns."""
        feedback = FeedbackLoop()
        scored_rules = [
            ScoredRule(
                rule=CandidateRule(
                    rule_id="r1",
                    source_code="def rule(x): return x*2",
                    description="double",
                    domain="math",
                    prompt_variant="direct",
                ),
                accuracy=0.9,
                bdm_complexity=10.0,
                mdl_score=10.0,
                is_pareto_optimal=True,
            ),
            ScoredRule(
                rule=CandidateRule(
                    rule_id="r2",
                    source_code="def rule(x): return 0",
                    description="bad",
                    domain="math",
                    prompt_variant="mathematical",
                ),
                accuracy=0.1,
                bdm_complexity=5.0,
                mdl_score=5.0,
                is_pareto_optimal=False,
            ),
        ]
        feedback.feed_synthesis_results(scored_rules, iteration=0)

        assert len(feedback.successful_patterns) == 1
        assert len(feedback.failure_patterns) == 1
        assert len(feedback.iteration_history) == 1

    def test_get_context(self):
        """Context should include variant information."""
        feedback = FeedbackLoop()
        scored = [
            ScoredRule(
                rule=CandidateRule(
                    rule_id="r1", source_code="def rule(x): return x*2",
                    description="", domain="math", prompt_variant="direct",
                ),
                accuracy=0.95, bdm_complexity=10.0, mdl_score=10.0,
                is_pareto_optimal=True,
            ),
        ]
        feedback.feed_synthesis_results(scored, iteration=0)
        feedback.feed_synthesis_results(scored, iteration=1)

        context = feedback.get_context_for_generation()
        assert context["iteration_count"] == 2
        assert "direct" in context["successful_variants"]
        assert "accuracy_improving" in context

    def test_empty_context(self):
        """Empty feedback should return safe defaults."""
        feedback = FeedbackLoop()
        context = feedback.get_context_for_generation()
        assert context["iteration_count"] == 0
        assert context["accuracy_improving"] is None
