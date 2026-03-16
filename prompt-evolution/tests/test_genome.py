"""Tests for genome module: creation, serialization, similarity."""

import json
import pytest

from src.genome.prompt_genome import PromptSection, PromptGenome
from src.genome.sections import (
    SECTION_ORDER,
    DEFAULT_SECTIONS,
    validate_sections,
    section_importance_weights,
    SECTION_IMPORTANCE_WEIGHTS,
)
from src.genome.serializer import serialize_genome, deserialize_genome
from src.genome.similarity import genome_similarity, section_similarity


class TestPromptSection:
    """Tests for PromptSection dataclass."""

    def test_creation(self):
        section = PromptSection(section_type="identity", content="You are a helper.")
        assert section.section_type == "identity"
        assert section.content == "You are a helper."
        assert section.token_count > 0
        assert section.is_mutable is True

    def test_token_estimation(self):
        section = PromptSection(section_type="test", content="one two three four five")
        assert section.token_count >= 5  # At least as many as words

    def test_empty_content(self):
        section = PromptSection(section_type="test", content="")
        assert section.token_count >= 1  # Minimum 1

    def test_copy(self):
        section = PromptSection(section_type="identity", content="Hello", is_mutable=False)
        copied = section.copy()
        assert copied.section_type == section.section_type
        assert copied.content == section.content
        assert copied.is_mutable == section.is_mutable
        # Modifying copy should not affect original
        copied.content = "Modified"
        assert section.content == "Hello"

    def test_explicit_token_count(self):
        section = PromptSection(section_type="test", content="hello", token_count=42)
        assert section.token_count == 42


class TestPromptGenome:
    """Tests for PromptGenome dataclass."""

    def test_creation(self):
        genome = PromptGenome()
        assert genome.genome_id is not None
        assert len(genome.genome_id) == 8
        assert genome.generation == 0
        assert genome.fitness == 0.0
        assert genome.sections == {}

    def test_set_and_get_section(self):
        genome = PromptGenome()
        genome.set_section("identity", "You are an AI.")
        section = genome.get_section("identity")
        assert section is not None
        assert section.content == "You are an AI."

    def test_get_missing_section(self):
        genome = PromptGenome()
        assert genome.get_section("nonexistent") is None

    def test_total_tokens(self):
        genome = PromptGenome()
        genome.set_section("identity", "word " * 10)
        genome.set_section("methodology", "word " * 20)
        tokens = genome.total_tokens()
        assert tokens > 0

    def test_to_system_prompt(self, sample_genome):
        prompt = sample_genome.to_system_prompt()
        assert "## identity" in prompt
        assert "## methodology" in prompt
        assert len(prompt) > 100

    def test_to_system_prompt_ordering(self):
        genome = PromptGenome()
        genome.set_section("methodology", "Method content")
        genome.set_section("identity", "Identity content")
        prompt = genome.to_system_prompt()
        # identity should come before methodology per SECTION_ORDER
        id_pos = prompt.index("## identity")
        meth_pos = prompt.index("## methodology")
        assert id_pos < meth_pos

    def test_copy(self, sample_genome):
        copied = sample_genome.copy()
        assert copied.genome_id != sample_genome.genome_id
        assert copied.generation == sample_genome.generation
        assert copied.fitness == sample_genome.fitness
        assert len(copied.sections) == len(sample_genome.sections)
        # Modifying copy should not affect original
        copied.set_section("identity", "Modified")
        assert sample_genome.sections["identity"].content != "Modified"

    def test_from_string(self):
        text = "## identity\nYou are a helper.\n\n## methodology\nSolve step by step."
        genome = PromptGenome.from_string(text)
        assert "identity" in genome.sections
        assert "methodology" in genome.sections
        assert genome.sections["identity"].content == "You are a helper."

    def test_from_string_no_headers(self):
        text = "Just a plain prompt with no sections."
        genome = PromptGenome.from_string(text)
        assert "identity" in genome.sections
        assert genome.sections["identity"].content == text

    def test_from_string_with_id(self):
        text = "## identity\nHello"
        genome = PromptGenome.from_string(text, genome_id="custom_id")
        assert genome.genome_id == "custom_id"

    def test_similarity_identical(self, sample_genome):
        sim = sample_genome.similarity(sample_genome)
        assert sim == pytest.approx(1.0)

    def test_similarity_different(self, sample_genome, sample_genome_b):
        sim = sample_genome.similarity(sample_genome_b)
        assert 0.0 < sim < 1.0

    def test_metadata(self):
        genome = PromptGenome(metadata={"key": "value"})
        assert genome.metadata["key"] == "value"

    def test_extra_sections_in_prompt(self):
        genome = PromptGenome()
        genome.set_section("identity", "I am AI")
        genome.set_section("custom_section", "Custom content")
        prompt = genome.to_system_prompt()
        assert "## custom_section" in prompt


class TestSections:
    """Tests for sections module."""

    def test_section_order_has_required(self):
        assert "identity" in SECTION_ORDER
        assert "methodology" in SECTION_ORDER
        assert "task_description" in SECTION_ORDER

    def test_default_sections_complete(self):
        for key in SECTION_ORDER:
            assert key in DEFAULT_SECTIONS

    def test_validate_sections_valid(self, sample_genome):
        errors = validate_sections(sample_genome.sections)
        assert len(errors) == 0

    def test_validate_sections_missing_required(self):
        genome = PromptGenome()
        genome.set_section("output_format", "Show answer.")
        errors = validate_sections(genome.sections)
        assert any("identity" in e for e in errors)
        assert any("task_description" in e for e in errors)
        assert any("methodology" in e for e in errors)

    def test_validate_sections_empty_required(self):
        genome = PromptGenome()
        genome.set_section("identity", "")
        genome.set_section("task_description", "OK")
        genome.set_section("methodology", "OK")
        errors = validate_sections(genome.sections)
        assert any("empty" in e.lower() for e in errors)

    def test_validate_sections_too_long(self):
        genome = PromptGenome()
        genome.set_section("identity", "word " * 500)
        genome.set_section("task_description", "OK")
        genome.set_section("methodology", "OK")
        errors = validate_sections(genome.sections)
        assert any("500 token" in e for e in errors)

    def test_section_importance_weights(self):
        weights = section_importance_weights()
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert "methodology" in weights


class TestSerializer:
    """Tests for genome serialization."""

    def test_serialize_roundtrip(self, sample_genome):
        json_str = serialize_genome(sample_genome)
        restored = deserialize_genome(json_str)
        assert restored.genome_id == sample_genome.genome_id
        assert restored.generation == sample_genome.generation
        assert restored.fitness == sample_genome.fitness
        assert len(restored.sections) == len(sample_genome.sections)

    def test_serialize_json_valid(self, sample_genome):
        json_str = serialize_genome(sample_genome)
        data = json.loads(json_str)
        assert "genome_id" in data
        assert "sections" in data

    def test_deserialize_preserves_sections(self, sample_genome):
        json_str = serialize_genome(sample_genome)
        restored = deserialize_genome(json_str)
        for name in sample_genome.sections:
            assert name in restored.sections
            assert (
                restored.sections[name].content
                == sample_genome.sections[name].content
            )

    def test_serialize_empty_genome(self):
        genome = PromptGenome(genome_id="empty")
        json_str = serialize_genome(genome)
        restored = deserialize_genome(json_str)
        assert restored.genome_id == "empty"
        assert len(restored.sections) == 0

    def test_deserialize_with_metadata(self):
        genome = PromptGenome(genome_id="meta", metadata={"foo": "bar"})
        json_str = serialize_genome(genome)
        restored = deserialize_genome(json_str)
        assert restored.metadata == {"foo": "bar"}


class TestSimilarity:
    """Tests for similarity metrics."""

    def test_section_similarity_identical(self):
        text = "Hello world this is a test"
        assert section_similarity(text, text) == 1.0

    def test_section_similarity_different(self):
        sim = section_similarity("hello world", "goodbye universe")
        assert sim == 0.0

    def test_section_similarity_partial(self):
        sim = section_similarity("hello world foo", "hello world bar")
        assert 0.0 < sim < 1.0

    def test_section_similarity_empty_both(self):
        assert section_similarity("", "") == 1.0

    def test_section_similarity_one_empty(self):
        assert section_similarity("hello", "") == 0.0
        assert section_similarity("", "hello") == 0.0

    def test_genome_similarity_identical(self, sample_genome):
        sim = genome_similarity(sample_genome, sample_genome)
        assert sim == pytest.approx(1.0)

    def test_genome_similarity_different(self, sample_genome, sample_genome_b):
        sim = genome_similarity(sample_genome, sample_genome_b)
        assert 0.0 < sim < 1.0

    def test_genome_similarity_empty(self):
        g1 = PromptGenome()
        g2 = PromptGenome()
        sim = genome_similarity(g1, g2)
        assert sim == 1.0

    def test_genome_similarity_one_has_extra_section(self, sample_genome):
        other = sample_genome.copy()
        other.set_section("custom", "Special content here")
        sim = genome_similarity(sample_genome, other)
        assert sim < 1.0
