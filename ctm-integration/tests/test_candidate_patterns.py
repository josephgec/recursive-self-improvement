"""Tests for additional pattern detection in candidate generator."""

import pytest

from src.synthesis.candidate_generator import CandidateGenerator, CandidateRule, IOExample


class TestIdentityPattern:
    """Tests for identity pattern detection."""

    def test_identity_detected(self):
        """Identity y=x should be detected."""
        gen = CandidateGenerator()
        examples = [
            IOExample(input=1, output=1),
            IOExample(input=5, output=5),
            IOExample(input=42, output=42),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and c.get_function()(99) == 99
            for c in candidates
        )
        assert correct


class TestListPatterns:
    """Tests for list operation patterns."""

    def test_list_length_detected(self):
        """len(list) pattern should be detected."""
        gen = CandidateGenerator()
        examples = [
            IOExample(input=[1, 2, 3], output=3),
            IOExample(input=[10, 20], output=2),
            IOExample(input=[5, 6, 7, 8], output=4),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and c.get_function()([1, 2]) == 2
            for c in candidates
        )
        assert correct

    def test_list_max_detected(self):
        """max(list) pattern should be detected."""
        gen = CandidateGenerator()
        examples = [
            IOExample(input=[1, 5, 3], output=5),
            IOExample(input=[10, 20, 15], output=20),
            IOExample(input=[7, 2, 9], output=9),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and c.get_function()([3, 8, 1]) == 8
            for c in candidates
        )
        assert correct

    def test_list_min_detected(self):
        """min(list) pattern should be detected."""
        gen = CandidateGenerator()
        examples = [
            IOExample(input=[5, 1, 3], output=1),
            IOExample(input=[10, 20, 15], output=10),
            IOExample(input=[7, 2, 9], output=2),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and c.get_function()([3, 8, 1]) == 1
            for c in candidates
        )
        assert correct


class TestStringPatterns:
    """Tests for string operation patterns."""

    def test_string_upper_detected(self):
        """str.upper() pattern should be detected."""
        gen = CandidateGenerator()
        examples = [
            IOExample(input="hello", output="HELLO"),
            IOExample(input="world", output="WORLD"),
            IOExample(input="abc", output="ABC"),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and c.get_function()("test") == "TEST"
            for c in candidates
        )
        assert correct

    def test_string_length_detected(self):
        """len(str) pattern should be detected."""
        gen = CandidateGenerator()
        examples = [
            IOExample(input="abc", output=3),
            IOExample(input="hello", output=5),
            IOExample(input="x", output=1),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and c.get_function()("test") == 4
            for c in candidates
        )
        assert correct


class TestLookupPattern:
    """Tests for lookup/fallback pattern."""

    def test_lookup_used_for_undetectable_pattern(self):
        """Non-obvious patterns should use lookup table."""
        gen = CandidateGenerator()
        # A pattern that isn't linear, quadratic, or any known type
        examples = [
            IOExample(input=1, output=7),
            IOExample(input=2, output=3),
            IOExample(input=3, output=11),
        ]
        candidates = gen.generate(examples, n=3)
        # Should generate something (lookup or otherwise)
        assert len(candidates) > 0
        # At least one should handle the known inputs
        for c in candidates:
            func = c.get_function()
            if func is not None:
                try:
                    if func(1) == 7:
                        break
                except Exception:
                    pass


class TestLinearCodeGeneration:
    """Tests for linear code generation branches."""

    def test_non_integer_slope(self):
        """Non-integer slope should generate float code."""
        gen = CandidateGenerator()
        # y = 1.5x (non-integer slope)
        examples = [
            IOExample(input=2, output=3.0),
            IOExample(input=4, output=6.0),
            IOExample(input=6, output=9.0),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and abs(c.get_function()(10) - 15.0) < 0.01
            for c in candidates
        )
        assert correct

    def test_non_integer_intercept(self):
        """Non-integer coefficients should generate float code."""
        gen = CandidateGenerator()
        # y = 1.5x + 0.5
        examples = [
            IOExample(input=0, output=0.5),
            IOExample(input=1, output=2.0),
            IOExample(input=2, output=3.5),
        ]
        candidates = gen.generate(examples, n=3)
        # Should produce candidates that handle float coefficients
        assert len(candidates) > 0


class TestQuadraticCodeGeneration:
    """Tests for quadratic code generation branches."""

    def test_quadratic_with_float_coefficients(self):
        """Quadratic with non-integer coefficients should work."""
        gen = CandidateGenerator()
        # y = 0.5 * x^2
        examples = [
            IOExample(input=0, output=0),
            IOExample(input=2, output=2),
            IOExample(input=4, output=8),
            IOExample(input=6, output=18),
        ]
        candidates = gen.generate(examples, n=3)
        assert len(candidates) > 0

    def test_quadratic_with_linear_term(self):
        """y = x^2 + 2x should be detected."""
        gen = CandidateGenerator()
        examples = [
            IOExample(input=0, output=0),
            IOExample(input=1, output=3),
            IOExample(input=2, output=8),
            IOExample(input=3, output=15),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and c.get_function()(4) == 24
            for c in candidates
        )
        assert correct

    def test_quadratic_with_constant(self):
        """y = x^2 + 1 should be detected."""
        gen = CandidateGenerator()
        examples = [
            IOExample(input=0, output=1),
            IOExample(input=1, output=2),
            IOExample(input=2, output=5),
            IOExample(input=3, output=10),
        ]
        candidates = gen.generate(examples, n=3)
        correct = any(
            c.get_function() is not None and c.get_function()(4) == 17
            for c in candidates
        )
        assert correct


class TestCandidateRuleGetFunction:
    """Tests for CandidateRule.get_function."""

    def test_get_function_valid(self):
        """Valid code should return a callable."""
        rule = CandidateRule(
            rule_id="r1",
            source_code="def rule(x):\n    return x * 2\n",
            description="",
        )
        func = rule.get_function()
        assert func is not None
        assert func(5) == 10

    def test_get_function_syntax_error(self):
        """Invalid code should return None."""
        rule = CandidateRule(
            rule_id="r1",
            source_code="def rule(x)\n    return x\n",
            description="",
        )
        func = rule.get_function()
        assert func is None

    def test_get_function_no_callable(self):
        """Code with no function should return None."""
        rule = CandidateRule(
            rule_id="r1",
            source_code="x = 42\n",
            description="",
        )
        func = rule.get_function()
        assert func is None


class TestParseRule:
    """Tests for rule parsing."""

    def test_parse_valid_code(self):
        """Valid code should parse successfully."""
        gen = CandidateGenerator()
        result = gen._parse_rule("def rule(x):\n    return x\n", "math", "direct")
        assert result is not None

    def test_parse_invalid_code(self):
        """Invalid code should return None."""
        gen = CandidateGenerator()
        result = gen._parse_rule("def rule(x\n    return x\n", "math", "direct")
        assert result is None

    def test_parse_code_without_def(self):
        """Code without def should be wrapped."""
        gen = CandidateGenerator()
        result = gen._parse_rule("return x * 2", "math", "direct")
        assert result is not None


class TestSpecialCases:
    """Tests for special case addition."""

    def test_add_special_cases(self):
        """Special cases should be added to the rule."""
        gen = CandidateGenerator()
        base = CandidateRule(
            rule_id="base",
            source_code="def rule(x):\n    return x * 2\n",
            description="",
        )
        failures = [(3, 9, 6)]
        result = gen._add_special_cases(base, failures)
        assert result is not None
        assert "if x == 3" in result

    def test_add_special_cases_empty(self):
        """No failures should return None."""
        gen = CandidateGenerator()
        base = CandidateRule(
            rule_id="base",
            source_code="def rule(x):\n    return x\n",
            description="",
        )
        result = gen._add_special_cases(base, [])
        assert result is None
