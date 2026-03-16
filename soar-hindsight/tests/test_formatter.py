"""Tests for the formatter module."""

import json
import os
import tempfile

import pytest

from src.synthesis.formatter import Formatter
from src.synthesis.synthesizer import TrainingPair


class TestFormatter:
    def test_format_openai(self, sample_training_pairs):
        formatter = Formatter(format_type="openai_jsonl")
        result = formatter.format_pair(sample_training_pairs[0])
        assert "messages" in result
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"
        assert result["messages"][1]["content"] == sample_training_pairs[0].prompt
        assert result["messages"][2]["content"] == sample_training_pairs[0].completion

    def test_format_huggingface(self, sample_training_pairs):
        formatter = Formatter(format_type="huggingface")
        result = formatter.format_pair(sample_training_pairs[0])
        assert "instruction" in result
        assert "output" in result
        assert "metadata" in result
        assert result["instruction"] == sample_training_pairs[0].prompt
        assert result["output"] == sample_training_pairs[0].completion

    def test_format_all(self, sample_training_pairs):
        formatter = Formatter(format_type="openai_jsonl")
        results = formatter.format_all(sample_training_pairs)
        assert len(results) == len(sample_training_pairs)

    def test_write_jsonl(self, sample_training_pairs):
        formatter = Formatter(format_type="openai_jsonl")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.jsonl")
            n = formatter.write_jsonl(sample_training_pairs, filepath)
            assert n == len(sample_training_pairs)
            assert os.path.exists(filepath)

            # Verify file content
            with open(filepath, "r") as f:
                lines = f.readlines()
            assert len(lines) == len(sample_training_pairs)
            for line in lines:
                data = json.loads(line)
                assert "messages" in data

    def test_read_jsonl(self, sample_training_pairs):
        formatter = Formatter(format_type="openai_jsonl")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.jsonl")
            formatter.write_jsonl(sample_training_pairs, filepath)
            items = formatter.read_jsonl(filepath)
            assert len(items) == len(sample_training_pairs)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            Formatter(format_type="invalid")

    def test_write_creates_directory(self, sample_training_pairs):
        formatter = Formatter(format_type="openai_jsonl")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "train.jsonl")
            n = formatter.write_jsonl(sample_training_pairs, filepath)
            assert n > 0
            assert os.path.exists(filepath)

    def test_huggingface_metadata(self, sample_training_pairs):
        formatter = Formatter(format_type="huggingface")
        result = formatter.format_pair(sample_training_pairs[0])
        assert result["metadata"]["pair_id"] == sample_training_pairs[0].pair_id
        assert result["metadata"]["strategy"] == sample_training_pairs[0].strategy
        assert result["metadata"]["quality_score"] == sample_training_pairs[0].quality_score

    def test_roundtrip(self, sample_training_pairs):
        formatter = Formatter(format_type="openai_jsonl")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.jsonl")
            formatter.write_jsonl(sample_training_pairs, filepath)
            items = formatter.read_jsonl(filepath)
            assert len(items) == len(sample_training_pairs)
            for item in items:
                assert item["messages"][1]["role"] == "user"
