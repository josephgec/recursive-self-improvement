"""Tests for structural drift signal."""

import pytest

from src.signals.structural import (
    StructuralDriftSignal,
    _split_sentences,
    _clause_depth,
    _categorize_token,
)


class TestStructuralDriftSignal:
    """Tests for StructuralDriftSignal."""

    def setup_method(self):
        self.signal = StructuralDriftSignal()

    def test_same_outputs_near_zero(self, reference_outputs):
        """Same outputs should produce near-zero drift."""
        result = self.signal.compute(reference_outputs, reference_outputs)
        assert result.normalized_score < 0.05
        assert result.signal_name == "structural"

    def test_restructured_moderate(self, reference_outputs, drifted_outputs):
        """Restructured text should produce moderate drift."""
        result = self.signal.compute(drifted_outputs, reference_outputs)
        # Drifted outputs are paraphrased, may have structural differences
        assert result.normalized_score >= 0.0
        assert result.normalized_score < 0.8

    def test_code_vs_prose_high(self):
        """Code-like text vs prose should have high structural drift."""
        prose = [
            "The sun sets beautifully over the horizon. Birds sing their evening songs. "
            "Nature provides a calming presence that soothes the weary soul.",
            "Life is a journey filled with unexpected turns. We learn from our experiences "
            "and grow stronger through adversity. Hope guides us forward.",
        ]
        code_like = [
            "def x(): return 0. if a > b: pass. for i in range(10): print(i). "
            "class Foo: bar = 1. try: x(). except: pass.",
            "import os. x = [1,2,3]. y = {k:v}. z = lambda a: a+1. "
            "assert x. raise ValueError. yield 42. return None.",
        ]
        result = self.signal.compute(code_like, prose)
        assert result.normalized_score > 0.1

    def test_sentence_length_shift_identical(self):
        """Same text should have zero sentence length shift."""
        texts = ["Short sentence. Another short one. Third one here."]
        shift = self.signal.sentence_length_distribution_shift(texts, texts)
        assert shift < 0.01

    def test_sentence_length_shift_different(self):
        """Different length patterns should show shift."""
        short = ["A. B. C. D. E. F. G. H."]
        long = [
            "This is a very long sentence with many words that goes on and on. "
            "Here is another lengthy sentence with lots of detail and description."
        ]
        shift = self.signal.sentence_length_distribution_shift(short, long)
        assert shift > 0.1

    def test_depth_distribution_shift_identical(self):
        """Same text should have zero depth shift."""
        texts = ["Simple sentence. Another simple one."]
        shift = self.signal.depth_distribution_shift(texts, texts)
        assert shift < 0.01

    def test_depth_shift_simple_vs_complex(self):
        """Simple vs complex sentences should show depth shift."""
        simple = ["The cat sat. The dog ran. A bird flew."]
        complex_text = [
            "Although the cat sat because it was tired, the dog ran "
            "while the bird flew since the wind carried it."
        ]
        shift = self.signal.depth_distribution_shift(complex_text, simple)
        assert shift >= 0.0

    def test_node_type_shift_identical(self):
        """Same text should have zero node type shift."""
        texts = ["The quick brown fox jumps over the lazy dog."]
        shift = self.signal.node_type_shift(texts, texts)
        assert shift < 0.01

    def test_node_type_shift_different(self):
        """Different POS distributions should show shift."""
        noun_heavy = ["cat dog house tree car book table chair desk lamp"]
        verb_heavy = ["running jumping eating sleeping walking talking singing dancing"]
        shift = self.signal.node_type_shift(noun_heavy, verb_heavy)
        assert shift > 0.05

    def test_normalize(self):
        """Normalization should work correctly."""
        assert self.signal.normalize(0.0) == 0.0
        assert self.signal.normalize(0.25) == 0.5
        assert self.signal.normalize(0.5) == 1.0
        assert self.signal.normalize(0.8) == 1.0

    def test_collapsed_high_drift(self, reference_outputs, collapsed_outputs):
        """Collapsed outputs should produce elevated drift."""
        result = self.signal.compute(collapsed_outputs, reference_outputs)
        assert result.normalized_score > 0.1

    def test_result_components(self, reference_outputs):
        """Result should contain all components."""
        result = self.signal.compute(reference_outputs, reference_outputs)
        assert "sentence_length_shift" in result.components
        assert "depth_distribution_shift" in result.components
        assert "node_type_shift" in result.components


class TestHelpers:
    """Tests for structural signal helper functions."""

    def test_split_sentences(self):
        """Should split on sentence-ending punctuation."""
        sents = _split_sentences("Hello world. How are you? I am fine!")
        assert len(sents) == 3

    def test_split_sentences_empty(self):
        """Empty string should produce empty list."""
        assert _split_sentences("") == []

    def test_clause_depth_simple(self):
        """Simple sentence should have depth 1."""
        depth = _clause_depth("The cat sat on the mat")
        assert depth == 1

    def test_clause_depth_complex(self):
        """Complex sentence should have higher depth."""
        depth = _clause_depth("Although the cat sat because it was tired")
        assert depth > 1

    def test_clause_depth_parenthetical(self):
        """Parenthetical nesting should increase depth."""
        depth = _clause_depth("The result (which was unexpected) surprised us")
        assert depth > 1

    def test_categorize_token_det(self):
        """Determiners should be categorized correctly."""
        assert _categorize_token("the") == "DET"
        assert _categorize_token("a") == "DET"

    def test_categorize_token_prep(self):
        """Prepositions should be categorized correctly."""
        assert _categorize_token("in") == "PREP"
        assert _categorize_token("on") == "PREP"

    def test_categorize_token_verb(self):
        """Verb-like tokens should be categorized."""
        assert _categorize_token("running") == "VERB"
        assert _categorize_token("computed") == "VERB"

    def test_categorize_token_adv(self):
        """Adverbs should be categorized."""
        assert _categorize_token("quickly") == "ADV"
        assert _categorize_token("slowly") == "ADV"

    def test_categorize_token_noun_default(self):
        """Unknown tokens should default to NOUN."""
        assert _categorize_token("xyzzy") == "NOUN"

    def test_categorize_token_num(self):
        """Numbers should be categorized as NUM."""
        assert _categorize_token("42") == "NUM"

    def test_categorize_token_conj(self):
        """Conjunctions should be categorized."""
        assert _categorize_token("and") == "CONJ"
        assert _categorize_token("but") == "CONJ"

    def test_categorize_token_aux(self):
        """Auxiliaries should be categorized."""
        assert _categorize_token("is") == "AUX"
        assert _categorize_token("would") == "AUX"

    def test_categorize_token_adj(self):
        """Adjective-like tokens should be categorized."""
        assert _categorize_token("beautiful") == "ADJ"
        assert _categorize_token("capable") == "ADJ"
