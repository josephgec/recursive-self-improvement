"""Tests for candidate rule generation with mock LLM."""

import pytest

from src.synthesis.candidate_generator import (
    CandidateGenerator,
    CandidateRule,
    IOExample,
)


class TestCandidateGeneration:
    """Tests for generating candidate rules."""

    def test_generate_returns_candidates(self, generator, double_examples):
        """generate() should return a list of CandidateRule objects."""
        candidates = generator.generate(double_examples, n=3)
        assert len(candidates) > 0
        assert all(isinstance(c, CandidateRule) for c in candidates)

    def test_generate_requested_count(self, generator, double_examples):
        """generate() should return approximately n candidates."""
        candidates = generator.generate(double_examples, n=5)
        assert len(candidates) == 5

    def test_generated_code_compiles(self, generator, double_examples):
        """All generated code should compile."""
        candidates = generator.generate(double_examples, n=5)
        for c in candidates:
            try:
                compile(c.source_code, "<test>", "exec")
            except SyntaxError:
                pytest.fail(f"Rule {c.rule_id} has syntax error: {c.source_code}")

    def test_generated_code_has_function(self, generator, double_examples):
        """Generated code should define a callable function."""
        candidates = generator.generate(double_examples, n=3)
        for c in candidates:
            func = c.get_function()
            assert func is not None, f"No callable in {c.rule_id}: {c.source_code}"

    def test_linear_pattern_detected(self, generator, linear_examples):
        """Mock LLM should detect linear pattern y = 2x + 1."""
        candidates = generator.generate(linear_examples, n=3)
        # At least one should get the pattern right
        correct = False
        for c in candidates:
            func = c.get_function()
            if func is not None:
                try:
                    if func(0) == 1 and func(1) == 3 and func(2) == 5:
                        correct = True
                        break
                except Exception:
                    pass
        assert correct, "No candidate detected the linear pattern"

    def test_sum_pattern_detected(self, generator, sum_examples):
        """Mock LLM should detect sum(list) pattern."""
        candidates = generator.generate(sum_examples, n=3)
        correct = False
        for c in candidates:
            func = c.get_function()
            if func is not None:
                try:
                    if func([1, 2, 3]) == 6:
                        correct = True
                        break
                except Exception:
                    pass
        assert correct, "No candidate detected the sum pattern"

    def test_diverse_prompt_variants(self, generator, double_examples):
        """Generated candidates should use diverse prompt variants."""
        candidates = generator.generate(double_examples, n=5)
        variants = set(c.prompt_variant for c in candidates)
        assert len(variants) >= 3  # At least 3 different variants

    def test_unique_rule_ids(self, generator, double_examples):
        """All generated rules should have unique IDs."""
        candidates = generator.generate(double_examples, n=5)
        ids = [c.rule_id for c in candidates]
        assert len(ids) == len(set(ids))


class TestRefinement:
    """Tests for rule refinement."""

    def test_generate_refinement(self, generator, double_examples):
        """Refinement should produce new candidates."""
        base = CandidateRule(
            rule_id="base",
            source_code="def rule(x):\n    return x + 1\n",
            description="Wrong rule",
            domain="math",
        )
        failures = [(1, 2, 2), (2, 4, 3)]  # (input, expected, actual)
        refined = generator.generate_refinement(base, failures, double_examples)
        assert len(refined) > 0

    def test_refinement_different_from_original(self, generator, double_examples):
        """Refined rules should differ from the original."""
        base = CandidateRule(
            rule_id="base",
            source_code="def rule(x):\n    return x + 1\n",
            description="Wrong rule",
            domain="math",
        )
        failures = [(1, 2, 2)]
        refined = generator.generate_refinement(base, failures, double_examples)
        # At least one should be different
        different = any(r.source_code != base.source_code for r in refined)
        assert different


class TestMockLLMPatterns:
    """Tests for specific pattern detection in mock LLM."""

    def test_quadratic_pattern(self, generator, quadratic_examples):
        """Mock LLM should detect quadratic pattern."""
        candidates = generator.generate(quadratic_examples, n=3)
        correct = False
        for c in candidates:
            func = c.get_function()
            if func is not None:
                try:
                    if func(2) == 4 and func(3) == 9:
                        correct = True
                        break
                except Exception:
                    pass
        assert correct, "No candidate detected the quadratic pattern"

    def test_constant_pattern(self, generator):
        """Mock LLM should detect constant output."""
        examples = [
            IOExample(input=1, output=42, domain="math"),
            IOExample(input=2, output=42, domain="math"),
            IOExample(input=100, output=42, domain="math"),
        ]
        candidates = generator.generate(examples, n=3)
        correct = False
        for c in candidates:
            func = c.get_function()
            if func is not None:
                try:
                    if func(999) == 42:
                        correct = True
                        break
                except Exception:
                    pass
        assert correct, "No candidate detected constant pattern"

    def test_string_reverse_pattern(self, generator):
        """Mock LLM should detect string reversal."""
        examples = [
            IOExample(input="abc", output="cba", domain="string"),
            IOExample(input="hello", output="olleh", domain="string"),
            IOExample(input="xy", output="yx", domain="string"),
        ]
        candidates = generator.generate(examples, n=3)
        correct = False
        for c in candidates:
            func = c.get_function()
            if func is not None:
                try:
                    if func("test") == "tset":
                        correct = True
                        break
                except Exception:
                    pass
        assert correct, "No candidate detected string reversal"
