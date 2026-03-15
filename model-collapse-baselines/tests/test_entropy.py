"""Tests for src.measurement.entropy.EntropyMeasurer."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.measurement.entropy import EntropyMeasurer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _FakeTokenizerOutput:
    """Mimics a HuggingFace BatchEncoding that supports .to() and **unpacking."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self._data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def to(self, device):
        return self  # no-op for CPU tests

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)


class _FakeModelOutput:
    """Mimics a HuggingFace model output with a .logits attribute."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


def _make_mock_model(vocab_size: int, uniform: bool = True):
    """Create a mock causal LM that returns controlled logits.

    Args:
        vocab_size: Vocabulary size.
        uniform: If True, return uniform logits (all zeros); otherwise
            return peaked logits (one-hot-ish).
    """
    fake_param = torch.zeros(1)

    class FakeModel:
        def eval(self):
            pass

        def parameters(self):
            return iter([fake_param])

        def __call__(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        input_ids = v
                        break

            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            if uniform:
                logits = torch.zeros(batch_size, seq_len, vocab_size)
            else:
                logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
                logits[:, :, 0] = 10.0

            return _FakeModelOutput(logits)

    return FakeModel()


def _make_mock_tokenizer(vocab_size: int = 100, max_seq_len: int = 32):
    """Create a mock tokenizer that does simple whitespace tokenization."""

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            max_len = kwargs.get("max_length", max_seq_len)
            all_ids = []
            all_masks = []
            for text in texts:
                words = text.split()
                ids = list(range(min(len(words), max_len)))
                if not ids:
                    ids = [0]
                all_ids.append(ids)
                all_masks.append([1] * len(ids))

            # Pad to same length.
            max_l = max(len(ids) for ids in all_ids)
            for i in range(len(all_ids)):
                pad_len = max_l - len(all_ids[i])
                all_ids[i] += [0] * pad_len
                all_masks[i] += [0] * pad_len

            return _FakeTokenizerOutput(
                input_ids=torch.tensor(all_ids, dtype=torch.long),
                attention_mask=torch.tensor(all_masks, dtype=torch.long),
            )

    return FakeTokenizer()


# ------------------------------------------------------------------
# Token entropy tests
# ------------------------------------------------------------------


class TestTokenEntropy:
    """Test per-position predictive entropy computation."""

    def test_uniform_distribution_entropy(self):
        """Uniform distribution should have entropy = log(vocab_size)."""
        vocab_size = 100
        model = _make_mock_model(vocab_size, uniform=True)
        tokenizer = _make_mock_tokenizer(vocab_size)
        measurer = EntropyMeasurer(batch_size=4)

        texts = ["word " * 10 for _ in range(8)]
        result = measurer.token_entropy(model, texts, tokenizer)

        expected = math.log(vocab_size)  # natural log
        assert abs(result.mean - expected) < 0.1, (
            f"Uniform entropy should be ~{expected:.2f}, got {result.mean:.2f}"
        )

    def test_peaked_distribution_has_low_entropy(self):
        """Peaked distribution should have near-zero entropy."""
        vocab_size = 100
        model = _make_mock_model(vocab_size, uniform=False)
        tokenizer = _make_mock_tokenizer(vocab_size)
        measurer = EntropyMeasurer(batch_size=4)

        texts = ["word " * 10 for _ in range(8)]
        result = measurer.token_entropy(model, texts, tokenizer)

        assert result.mean < 1.0, (
            f"Peaked distribution entropy should be low, got {result.mean:.2f}"
        )

    def test_low_entropy_fraction(self):
        """Peaked model should have high low_entropy_fraction."""
        vocab_size = 100
        model = _make_mock_model(vocab_size, uniform=False)
        tokenizer = _make_mock_tokenizer(vocab_size)
        measurer = EntropyMeasurer(batch_size=4)

        texts = ["word " * 10 for _ in range(8)]
        result = measurer.token_entropy(model, texts, tokenizer)

        assert result.low_entropy_fraction > 0.9, (
            f"Most positions should have low entropy, got {result.low_entropy_fraction:.2f}"
        )

    def test_result_has_histogram(self):
        """Result should include an entropy histogram."""
        vocab_size = 50
        model = _make_mock_model(vocab_size, uniform=True)
        tokenizer = _make_mock_tokenizer(vocab_size)
        measurer = EntropyMeasurer(batch_size=4)

        texts = ["word " * 5 for _ in range(4)]
        result = measurer.token_entropy(model, texts, tokenizer)

        assert isinstance(result.entropy_histogram, np.ndarray)
        assert result.entropy_histogram.sum() > 0


# ------------------------------------------------------------------
# Sequence entropy tests
# ------------------------------------------------------------------


class TestSequenceEntropy:
    """Test sequence-level n-gram entropy."""

    def test_repetitive_text_low_entropy(self):
        """Highly repetitive text should have low entropy."""
        measurer = EntropyMeasurer()
        repetitive = ["the the the the the the the the the the"] * 20
        result = measurer.sequence_entropy(repetitive)

        assert result.unigram < 0.5  # only one word
        assert result.bigram < 0.5

    def test_diverse_text_higher_entropy(self):
        """Diverse text should have higher entropy than repetitive text."""
        measurer = EntropyMeasurer()

        repetitive = ["cat cat cat cat cat cat cat cat"] * 10
        diverse = [
            "the cat sat on the mat by the door",
            "a quick brown fox jumps over the lazy dog",
            "she sells sea shells by the sea shore daily",
            "how much wood would a woodchuck chuck today",
            "peter piper picked a peck of pickled peppers",
        ] * 2

        rep_result = measurer.sequence_entropy(repetitive)
        div_result = measurer.sequence_entropy(diverse)

        assert div_result.unigram > rep_result.unigram
        assert div_result.bigram > rep_result.bigram

    def test_trigram_entropy_computed(self):
        """Trigram entropy should be computable."""
        measurer = EntropyMeasurer()
        texts = ["one two three four five six seven eight nine ten"] * 5
        result = measurer.sequence_entropy(texts)

        assert result.trigram > 0

    def test_empty_texts(self):
        """Empty texts should return zero entropy."""
        measurer = EntropyMeasurer()
        result = measurer.sequence_entropy([])
        assert result.unigram == 0.0
        assert result.bigram == 0.0
        assert result.trigram == 0.0

    def test_single_word(self):
        """Single word repeated should give zero unigram entropy."""
        measurer = EntropyMeasurer()
        result = measurer.sequence_entropy(["hello"] * 10)
        assert result.unigram == 0.0


# ------------------------------------------------------------------
# Conditional entropy trajectory tests
# ------------------------------------------------------------------


class TestConditionalEntropyTrajectory:
    """Test per-position averaged entropy trajectory."""

    def test_trajectory_shape(self):
        """Trajectory should produce an array of position entropies."""
        vocab_size = 50
        model = _make_mock_model(vocab_size, uniform=True)
        tokenizer = _make_mock_tokenizer(vocab_size)
        measurer = EntropyMeasurer(batch_size=4)

        texts = ["word " * 8 for _ in range(4)]
        result = measurer.conditional_entropy_trajectory(
            model, texts, tokenizer, max_length=32
        )

        assert len(result.position_entropies) > 0
        assert result.num_texts == 4

    def test_uniform_trajectory_is_flat(self):
        """Uniform model should give roughly constant entropy at each position."""
        vocab_size = 50
        model = _make_mock_model(vocab_size, uniform=True)
        tokenizer = _make_mock_tokenizer(vocab_size)
        measurer = EntropyMeasurer(batch_size=4)

        texts = ["word " * 8 for _ in range(8)]
        result = measurer.conditional_entropy_trajectory(
            model, texts, tokenizer, max_length=32
        )

        # All non-zero positions should be approximately equal.
        nonzero = result.position_entropies[result.position_entropies > 0]
        if len(nonzero) > 1:
            assert np.std(nonzero) < 0.5, (
                f"Uniform model trajectory should be flat, std={np.std(nonzero):.2f}"
            )
