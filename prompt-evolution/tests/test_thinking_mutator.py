"""Tests for thinking-model mutator."""

import pytest

from src.genome.prompt_genome import PromptGenome
from src.genome.sections import DEFAULT_SECTIONS
from src.operators.thinking_mutator import ThinkingMutator, WeaknessAnalysis


class TestWeaknessAnalysis:
    """Tests for weakness analysis."""

    def test_dataclass_creation(self):
        wa = WeaknessAnalysis(
            weakest_section="methodology",
            weakness_score=0.3,
            details="Low quality content",
        )
        assert wa.weakest_section == "methodology"
        assert wa.weakness_score == 0.3

    def test_with_suggestions(self):
        wa = WeaknessAnalysis(
            weakest_section="identity",
            weakness_score=0.5,
            details="Vague identity",
            suggested_improvements=["Be more specific"],
        )
        assert len(wa.suggested_improvements) == 1


class TestThinkingMutator:
    """Tests for ThinkingMutator."""

    def test_mutate_returns_new_genome(self, mock_thinking_llm, sample_genome):
        mutator = ThinkingMutator(mock_thinking_llm)
        mutated = mutator.mutate(sample_genome)

        assert mutated.genome_id != sample_genome.genome_id
        assert mutated.generation == sample_genome.generation + 1
        assert mutated.parent_ids == [sample_genome.genome_id]

    def test_mutate_resets_fitness(self, mock_thinking_llm, sample_genome):
        mutator = ThinkingMutator(mock_thinking_llm)
        sample_genome.fitness = 0.8
        mutated = mutator.mutate(sample_genome)
        assert mutated.fitness == 0.0

    def test_analyze_weakness_heuristic(self, mock_thinking_llm, sample_genome):
        mutator = ThinkingMutator(mock_thinking_llm)
        weakness = mutator.analyze_weakness(sample_genome)

        assert weakness.weakest_section in sample_genome.sections
        assert weakness.weakness_score >= 0.0
        assert len(weakness.details) > 0

    def test_analyze_weakness_with_section_scores(
        self, mock_thinking_llm, sample_genome
    ):
        mutator = ThinkingMutator(mock_thinking_llm)
        fitness_details = {
            "section_scores": {
                "identity": 0.8,
                "methodology": 0.3,
                "reasoning_style": 0.6,
            }
        }
        weakness = mutator.analyze_weakness(sample_genome, fitness_details)
        assert weakness.weakest_section == "methodology"

    def test_mutate_targets_weakest_section(self, mock_thinking_llm, sample_genome):
        mutator = ThinkingMutator(mock_thinking_llm)
        fitness_details = {
            "section_scores": {
                "identity": 0.9,
                "methodology": 0.2,
                "reasoning_style": 0.7,
                "task_description": 0.8,
            }
        }
        mutated = mutator.mutate(sample_genome, fitness_details)
        assert "mutate_methodology" in mutated.operator

    def test_mutation_scope_single_section(self, mock_thinking_llm, sample_genome):
        mutator = ThinkingMutator(mock_thinking_llm)
        original_sections = {
            k: v.content for k, v in sample_genome.sections.items()
        }
        mutated = mutator.mutate(sample_genome)

        # At least one section should be different
        changed = 0
        for name in original_sections:
            if name in mutated.sections:
                if mutated.sections[name].content != original_sections[name]:
                    changed += 1

        assert changed >= 1

    def test_mutate_preserves_immutable(self, mock_thinking_llm):
        genome = PromptGenome(genome_id="immut_test")
        genome.set_section("identity", "Immutable identity", is_mutable=False)
        genome.set_section("methodology", "Mutable methodology")
        genome.fitness = 0.5

        mutator = ThinkingMutator(mock_thinking_llm)
        mutated = mutator.mutate(genome)

        # Immutable section should remain unchanged
        assert mutated.sections["identity"].content == "Immutable identity"

    def test_analyze_weakness_no_mutable_sections(self, mock_thinking_llm):
        genome = PromptGenome(genome_id="no_mut")
        genome.set_section("identity", "Fixed", is_mutable=False)

        mutator = ThinkingMutator(mock_thinking_llm)
        weakness = mutator.analyze_weakness(genome)
        assert weakness.weakest_section == "methodology"  # Default fallback

    def test_mutate_with_empty_examples_section(self, mock_thinking_llm):
        genome = PromptGenome(genome_id="empty_ex")
        genome.set_section("identity", "AI")
        genome.set_section("task_description", "Solve")
        genome.set_section("methodology", "Steps")
        genome.set_section("examples", "")
        genome.fitness = 0.4

        mutator = ThinkingMutator(mock_thinking_llm)
        weakness = mutator.analyze_weakness(genome)
        # Empty examples should score low
        assert weakness.weakest_section in genome.sections
