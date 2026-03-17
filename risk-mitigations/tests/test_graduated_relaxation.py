"""Tests for GraduatedRelaxation - propose, apply, revert, limits, safety."""

import pytest
from src.constraints.graduated_relaxation import GraduatedRelaxation, SAFETY_CONSTRAINTS


class TestPropose:
    """Tests for constraint relaxation proposals."""

    def test_propose_returns_proposal(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("quality_threshold", 0.90)
        proposal = gr.propose_relaxation("quality_threshold")
        assert proposal.approved is True
        assert proposal.current_value == pytest.approx(0.90)
        assert proposal.proposed_value == pytest.approx(0.92)

    def test_propose_step_size_is_2pp(self):
        gr = GraduatedRelaxation(step_size_pp=2)
        gr.set_constraint("quality_threshold", 0.90)
        proposal = gr.propose_relaxation("quality_threshold")
        assert proposal.proposed_value - proposal.current_value == pytest.approx(0.02)

    def test_propose_unknown_constraint_raises(self):
        gr = GraduatedRelaxation()
        with pytest.raises(KeyError):
            gr.propose_relaxation("nonexistent")


class TestApply:
    """Tests for applying relaxation proposals."""

    def test_apply_changes_value(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("quality_threshold", 0.90)
        proposal = gr.propose_relaxation("quality_threshold")
        gr.apply_relaxation(proposal)
        assert gr.get_constraint_value("quality_threshold") == pytest.approx(0.92)

    def test_apply_rejected_proposal_returns_false(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("no_harmful_output", 1.0)
        proposal = gr.propose_relaxation("no_harmful_output")
        assert gr.apply_relaxation(proposal) is False


class TestRevert:
    """Tests for reverting relaxations."""

    def test_revert_restores_original(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("quality_threshold", 0.90)
        proposal = gr.propose_relaxation("quality_threshold")
        gr.apply_relaxation(proposal)
        original = gr.revert("quality_threshold")
        assert original == pytest.approx(0.90)
        assert gr.get_constraint_value("quality_threshold") == pytest.approx(0.90)

    def test_revert_resets_step_count(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("quality_threshold", 0.90)
        for _ in range(3):
            p = gr.propose_relaxation("quality_threshold")
            gr.apply_relaxation(p)
        assert gr.can_relax_further("quality_threshold") is False
        gr.revert("quality_threshold")
        assert gr.can_relax_further("quality_threshold") is True

    def test_revert_unknown_raises(self):
        gr = GraduatedRelaxation()
        with pytest.raises(KeyError):
            gr.revert("nonexistent")


class TestMaxSteps:
    """Tests for maximum relaxation steps."""

    def test_max_3_steps(self):
        gr = GraduatedRelaxation(max_steps=3)
        gr.set_constraint("quality_threshold", 0.90)

        for i in range(3):
            p = gr.propose_relaxation("quality_threshold")
            assert p.approved is True
            gr.apply_relaxation(p)

        # 4th step should be rejected
        p = gr.propose_relaxation("quality_threshold")
        assert p.approved is False

    def test_can_relax_further_after_max(self):
        gr = GraduatedRelaxation(max_steps=3)
        gr.set_constraint("quality_threshold", 0.90)
        for _ in range(3):
            p = gr.propose_relaxation("quality_threshold")
            gr.apply_relaxation(p)
        assert gr.can_relax_further("quality_threshold") is False

    def test_final_value_after_3_steps(self):
        gr = GraduatedRelaxation(max_steps=3, step_size_pp=2)
        gr.set_constraint("quality_threshold", 0.90)
        for _ in range(3):
            p = gr.propose_relaxation("quality_threshold")
            gr.apply_relaxation(p)
        # 0.90 + 3 * 0.02 = 0.96
        assert gr.get_constraint_value("quality_threshold") == pytest.approx(0.96)


class TestSafetyConstraints:
    """Tests that safety constraints can NEVER be relaxed."""

    def test_safety_constraint_rejected(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("no_harmful_output", 1.0)
        proposal = gr.propose_relaxation("no_harmful_output")
        assert proposal.approved is False
        assert proposal.is_safety_constraint is True

    def test_all_safety_constraints_rejected(self):
        gr = GraduatedRelaxation()
        for name in SAFETY_CONSTRAINTS:
            gr.set_constraint(name, 1.0)
            proposal = gr.propose_relaxation(name)
            assert proposal.approved is False, f"Safety constraint '{name}' was approved!"

    def test_safety_cannot_relax_further(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("maintain_alignment", 1.0)
        assert gr.can_relax_further("maintain_alignment") is False

    def test_safety_value_unchanged(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("no_harmful_output", 1.0)
        proposal = gr.propose_relaxation("no_harmful_output")
        gr.apply_relaxation(proposal)
        assert gr.get_constraint_value("no_harmful_output") == pytest.approx(1.0)


class TestHistoryAndState:
    """Tests for history tracking and state management."""

    def test_history_tracked(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("quality_threshold", 0.90)
        gr.propose_relaxation("quality_threshold")
        assert len(gr.get_history()) == 1

    def test_get_all_constraints(self):
        gr = GraduatedRelaxation()
        gr.set_constraint("a", 0.5)
        gr.set_constraint("b", 0.7)
        constraints = gr.get_all_constraints()
        assert len(constraints) == 2
        assert constraints["a"] == pytest.approx(0.5)
