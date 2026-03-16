"""Tests for quality filtering and deduplication."""

import pytest

from src.synthesis.quality_filter import QualityFilter
from src.synthesis.deduplicator import Deduplicator
from src.synthesis.synthesizer import TrainingPair


class TestQualityFilter:
    def test_passes_good_pairs(self, sample_training_pairs):
        qf = QualityFilter(
            min_prompt_tokens=5,
            max_prompt_tokens=1000,
            min_completion_tokens=5,
            max_completion_tokens=1000,
            min_quality_score=0.3,
        )
        result = qf.filter(sample_training_pairs)
        assert len(result) == len(sample_training_pairs)

    def test_rejects_low_quality(self, sample_training_pairs):
        qf = QualityFilter(min_quality_score=0.95)
        result = qf.filter(sample_training_pairs)
        assert len(result) < len(sample_training_pairs)

    def test_rejects_empty_prompt(self):
        pair = TrainingPair(prompt="", completion="code", prompt_tokens=0, completion_tokens=10)
        qf = QualityFilter(min_prompt_tokens=0, min_completion_tokens=0, min_quality_score=0.0)
        reasons = qf.check(pair)
        assert "empty_prompt" in reasons

    def test_rejects_empty_completion(self):
        pair = TrainingPair(prompt="solve this", completion="", prompt_tokens=10, completion_tokens=0)
        qf = QualityFilter(min_prompt_tokens=0, min_completion_tokens=0, min_quality_score=0.0)
        reasons = qf.check(pair)
        assert "empty_completion" in reasons

    def test_rejects_prompt_too_short(self):
        pair = TrainingPair(prompt="hi", completion="code here", prompt_tokens=2, completion_tokens=10, quality_score=0.5)
        qf = QualityFilter(min_prompt_tokens=5)
        reasons = qf.check(pair)
        assert any("prompt_too_short" in r for r in reasons)

    def test_rejects_prompt_too_long(self):
        pair = TrainingPair(prompt="x" * 10000, completion="code", prompt_tokens=5000, completion_tokens=10, quality_score=0.5)
        qf = QualityFilter(max_prompt_tokens=100)
        reasons = qf.check(pair)
        assert any("prompt_too_long" in r for r in reasons)

    def test_rejects_completion_too_short(self):
        pair = TrainingPair(prompt="solve this problem", completion="x", prompt_tokens=20, completion_tokens=1, quality_score=0.5)
        qf = QualityFilter(min_completion_tokens=5)
        reasons = qf.check(pair)
        assert any("completion_too_short" in r for r in reasons)

    def test_rejects_completion_too_long(self):
        pair = TrainingPair(prompt="solve", completion="x" * 10000, prompt_tokens=10, completion_tokens=5000, quality_score=0.5)
        qf = QualityFilter(max_completion_tokens=100)
        reasons = qf.check(pair)
        assert any("completion_too_long" in r for r in reasons)

    def test_rejects_identical(self):
        pair = TrainingPair(prompt="same text", completion="same text", prompt_tokens=10, completion_tokens=10, quality_score=0.5)
        qf = QualityFilter(min_prompt_tokens=0, min_completion_tokens=0, min_quality_score=0.0)
        reasons = qf.check(pair)
        assert "prompt_completion_identical" in reasons

    def test_check_returns_empty_for_good(self, sample_training_pairs):
        qf = QualityFilter(
            min_prompt_tokens=5,
            max_prompt_tokens=1000,
            min_completion_tokens=5,
            max_completion_tokens=1000,
            min_quality_score=0.3,
        )
        reasons = qf.check(sample_training_pairs[0])
        assert reasons == []

    def test_rejected_property(self, sample_training_pairs):
        qf = QualityFilter(min_quality_score=0.95)
        qf.filter(sample_training_pairs)
        assert len(qf.rejected) > 0
        for rej in qf.rejected:
            assert "pair_id" in rej
            assert "reasons" in rej

    def test_summary(self, sample_training_pairs):
        qf = QualityFilter(min_quality_score=0.95)
        qf.filter(sample_training_pairs)
        s = qf.summary()
        assert "rejected_count" in s
        assert "rejection_reasons" in s
        assert s["rejected_count"] > 0

    def test_rejects_low_quality_score(self):
        pair = TrainingPair(prompt="solve", completion="code", prompt_tokens=10, completion_tokens=10, quality_score=0.1)
        qf = QualityFilter(min_quality_score=0.5)
        reasons = qf.check(pair)
        assert any("low_quality" in r for r in reasons)


class TestDeduplicator:
    def test_removes_exact_duplicates(self):
        pairs = [
            TrainingPair(pair_id="a", completion="def add(a, b): return a + b", quality_score=0.9),
            TrainingPair(pair_id="b", completion="def add(a, b): return a + b", quality_score=0.8),
        ]
        dedup = Deduplicator(similarity_threshold=0.85)
        result = dedup.deduplicate(pairs)
        assert len(result) == 1
        assert result[0].pair_id == "a"  # keeps highest quality

    def test_keeps_distinct(self):
        pairs = [
            TrainingPair(pair_id="a", completion="def sort(lst): return sorted(lst)", quality_score=0.9),
            TrainingPair(pair_id="b", completion="def fibonacci(n): return fib_helper(n, 0, 1)", quality_score=0.8),
        ]
        dedup = Deduplicator(similarity_threshold=0.85)
        result = dedup.deduplicate(pairs)
        assert len(result) == 2

    def test_removed_count(self):
        pairs = [
            TrainingPair(pair_id="a", completion="same code here", quality_score=0.9),
            TrainingPair(pair_id="b", completion="same code here", quality_score=0.8),
            TrainingPair(pair_id="c", completion="different code entirely", quality_score=0.7),
        ]
        dedup = Deduplicator(similarity_threshold=0.85)
        result = dedup.deduplicate(pairs)
        assert dedup.removed_count == 1
        assert len(result) == 2

    def test_empty_input(self):
        dedup = Deduplicator()
        result = dedup.deduplicate([])
        assert result == []
        assert dedup.removed_count == 0

    def test_high_threshold_keeps_more(self):
        pairs = [
            TrainingPair(pair_id="a", completion="def add(a, b): return a + b", quality_score=0.9),
            TrainingPair(pair_id="b", completion="def add(x, y): return x + y", quality_score=0.8),
        ]
        dedup = Deduplicator(similarity_threshold=0.99)
        result = dedup.deduplicate(pairs)
        assert len(result) == 2

    def test_summary(self):
        pairs = [
            TrainingPair(pair_id="a", completion="abc", quality_score=0.9),
            TrainingPair(pair_id="b", completion="abc", quality_score=0.8),
        ]
        dedup = Deduplicator(similarity_threshold=0.85)
        dedup.deduplicate(pairs)
        s = dedup.summary()
        assert s["removed_count"] == 1

    def test_jaccard_similarity(self):
        assert Deduplicator._jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
        assert Deduplicator._jaccard_similarity({"a"}, {"b"}) == 0.0
        assert Deduplicator._jaccard_similarity(set(), set()) == 1.0
        assert Deduplicator._jaccard_similarity(set(), {"a"}) == 0.0

    def test_ngrams_short_text(self):
        dedup = Deduplicator(ngram_size=3)
        ngrams = dedup._compute_ngrams("ab")
        assert ngrams == {"ab"}

    def test_keeps_best_quality_among_duplicates(self):
        pairs = [
            TrainingPair(pair_id="low", completion="same code same code", quality_score=0.3),
            TrainingPair(pair_id="high", completion="same code same code", quality_score=0.95),
            TrainingPair(pair_id="mid", completion="same code same code", quality_score=0.6),
        ]
        dedup = Deduplicator(similarity_threshold=0.85)
        result = dedup.deduplicate(pairs)
        assert len(result) == 1
        assert result[0].pair_id == "high"
