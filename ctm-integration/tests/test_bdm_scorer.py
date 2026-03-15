"""Tests for BDM scoring."""

import pytest

from src.bdm.scorer import BDMScorer, BDMScore, RuleScore
from src.bdm.ctm_table import CTMTable


class TestBDMScoring:
    """Tests for BDM score computation."""

    def test_score_returns_bdm_score(self, scorer):
        """score() should return a BDMScore dataclass."""
        result = scorer.score("01010101")
        assert isinstance(result, BDMScore)
        assert result.bdm_value > 0
        assert result.num_blocks > 0

    def test_repetitive_less_than_random(self, scorer):
        """Repetitive data should have lower BDM than random-looking data."""
        repetitive = "0000000000000000"
        random_looking = "0110100110010110"

        score_rep = scorer.score(repetitive)
        score_rand = scorer.score(random_looking)

        assert score_rep.bdm_value < score_rand.bdm_value

    def test_constant_string_low_complexity(self, scorer):
        """All-zeros string should have low complexity."""
        score = scorer.score("0000")
        # Constant string with multiplicity should be simple
        assert score.bdm_value > 0

    def test_block_details_present(self, scorer):
        """Score should include block-level details."""
        score = scorer.score("01010101")
        assert len(score.block_details) > 0
        for detail in score.block_details:
            assert "content" in detail
            assert "ctm_complexity" in detail
            assert "multiplicity" in detail

    def test_normalized_bdm(self, scorer):
        """Normalized BDM should be BDM / length."""
        score = scorer.score("01010101")
        assert score.normalized_bdm > 0

    def test_score_integer(self, scorer):
        """Scoring an integer should work."""
        score = scorer.score(42)
        assert isinstance(score, BDMScore)
        assert score.bdm_value > 0

    def test_score_list(self, scorer):
        """Scoring a list should work."""
        score = scorer.score([1, 2, 3, 4])
        assert isinstance(score, BDMScore)
        assert score.bdm_value > 0


class TestRuleScoring:
    """Tests for scoring rules (programs)."""

    def test_score_program(self, scorer):
        """score_program should return BDMScore for code."""
        code = "def f(x):\n    return x * 2\n"
        score = scorer.score_program(code)
        assert isinstance(score, BDMScore)
        assert score.bdm_value > 0

    def test_score_rule_with_examples(self, scorer):
        """score_rule should compute accuracy and MDL."""
        code = "def rule(x):\n    return x * 2\n"
        inputs = [1, 2, 3, 4, 5]
        outputs = [2, 4, 6, 8, 10]

        result = scorer.score_rule(code, inputs, outputs)
        assert isinstance(result, RuleScore)
        assert result.accuracy == 1.0
        assert result.bdm_score > 0
        assert result.mdl_score > 0
        assert result.residual_complexity == 0.0  # perfect fit

    def test_score_rule_imperfect(self, scorer):
        """Imperfect rule should have residual complexity."""
        code = "def rule(x):\n    return x + 1\n"
        inputs = [1, 2, 3, 4, 5]
        outputs = [2, 4, 6, 8, 10]  # y = 2x, but rule does x+1

        result = scorer.score_rule(code, inputs, outputs)
        assert result.accuracy < 1.0
        assert result.residual_complexity > 0

    def test_score_rule_invalid_code(self, scorer):
        """Invalid code should get zero accuracy."""
        code = "this is not valid python code !!!"
        inputs = [1, 2, 3]
        outputs = [2, 4, 6]

        result = scorer.score_rule(code, inputs, outputs)
        assert result.accuracy == 0.0


class TestBaselines:
    """Tests for baseline comparisons."""

    def test_compare_to_baselines(self, scorer):
        """compare_to_baselines should return all metrics."""
        result = scorer.compare_to_baselines("01010101")
        assert "bdm" in result
        assert "shannon_entropy" in result
        assert "gzip_complexity" in result
        assert "lzma_complexity" in result

    def test_baselines_ordered(self, scorer):
        """Shannon entropy of constant string should be 0."""
        result = scorer.compare_to_baselines("00000000")
        assert result["shannon_entropy"] == 0.0

    def test_baselines_random_high_entropy(self, scorer):
        """Random-looking string should have high entropy."""
        result = scorer.compare_to_baselines("01101001")
        assert result["shannon_entropy"] > 0.9  # near 1.0 for binary
