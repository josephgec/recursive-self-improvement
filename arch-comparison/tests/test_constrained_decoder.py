"""Tests for constrained decoder: arithmetic constraint, computation detection."""

from __future__ import annotations

import pytest

from src.integrative.constrained_decoder import (
    Constraint,
    ConstrainedDecoder,
    ConstrainedOutput,
)
from src.integrative.logical_loss import LogicalLoss


# ── ConstrainedDecoder tests ──

class TestConstrainedDecoder:
    def test_generate_basic(self):
        decoder = ConstrainedDecoder()
        output = decoder.generate("What is 3 + 4?")
        assert isinstance(output, ConstrainedOutput)
        assert output.text != ""

    def test_detect_computation_context_arithmetic(self):
        decoder = ConstrainedDecoder()
        assert decoder._detect_computation_context("What is 3 + 4?")
        assert decoder._detect_computation_context("Calculate 5 * 6")
        assert decoder._detect_computation_context("Compute the sum")

    def test_detect_computation_context_negative(self):
        decoder = ConstrainedDecoder()
        assert not decoder._detect_computation_context("The sky is blue")
        assert not decoder._detect_computation_context("Hello world")

    def test_detect_computation_context_with_numbers(self):
        decoder = ConstrainedDecoder()
        assert decoder._detect_computation_context("3 + 4")

    def test_detect_computation_keywords(self):
        decoder = ConstrainedDecoder()
        assert decoder._detect_computation_context("solve this equation")
        assert decoder._detect_computation_context("evaluate the expression")
        assert decoder._detect_computation_context("find the result")

    def test_arithmetic_constraint_correct(self):
        decoder = ConstrainedDecoder()
        text = "The result is 3 + 4 = 7"
        corrected, applied, violations = decoder._apply_arithmetic_constraint(
            text, Constraint(constraint_type="arithmetic", pattern="")
        )
        assert applied
        assert violations == 0
        assert "3 + 4 = 7" in corrected

    def test_arithmetic_constraint_incorrect(self):
        decoder = ConstrainedDecoder()
        text = "The result is 3 + 4 = 8"
        corrected, applied, violations = decoder._apply_arithmetic_constraint(
            text, Constraint(constraint_type="arithmetic", pattern="")
        )
        assert applied
        assert violations == 1
        assert "3 + 4 = 7" in corrected

    def test_arithmetic_constraint_multiplication(self):
        decoder = ConstrainedDecoder()
        text = "5 * 6 = 31"
        corrected, applied, violations = decoder._apply_arithmetic_constraint(
            text, Constraint(constraint_type="arithmetic", pattern="")
        )
        assert applied
        assert violations == 1
        assert "5 * 6 = 30" in corrected

    def test_arithmetic_constraint_subtraction(self):
        decoder = ConstrainedDecoder()
        text = "10 - 3 = 6"
        corrected, applied, violations = decoder._apply_arithmetic_constraint(
            text, Constraint(constraint_type="arithmetic", pattern="")
        )
        assert applied
        assert violations == 1
        assert "10 - 3 = 7" in corrected

    def test_arithmetic_constraint_no_match(self):
        decoder = ConstrainedDecoder()
        text = "No arithmetic here"
        corrected, applied, violations = decoder._apply_arithmetic_constraint(
            text, Constraint(constraint_type="arithmetic", pattern="")
        )
        assert not applied
        assert violations == 0

    def test_logical_constraint_no_contradiction(self):
        decoder = ConstrainedDecoder()
        text = "A is true and B is false"
        corrected, applied, violations = decoder._apply_logical_constraint(
            text, Constraint(constraint_type="logical", pattern="")
        )
        assert not applied  # No contradiction
        assert violations == 0

    def test_logical_constraint_contradiction(self):
        decoder = ConstrainedDecoder()
        text = "A is true and A is false"
        corrected, applied, violations = decoder._apply_logical_constraint(
            text, Constraint(constraint_type="logical", pattern="")
        )
        assert applied
        assert violations >= 1

    def test_generate_with_constraints(self):
        def model(prompt):
            return "3 + 4 = 8"

        decoder = ConstrainedDecoder(model=model)
        decoder.add_constraint(Constraint(
            constraint_type="arithmetic",
            pattern=r"\d+\s*[\+\-\*]\s*\d+\s*=\s*\d+",
        ))
        output = decoder.generate("What is 3 + 4?")
        assert "3 + 4 = 7" in output.text
        assert output.constraint_violations >= 1

    def test_confidence_decreases_with_violations(self):
        def model(prompt):
            return "3 + 4 = 8 and 5 + 6 = 13"

        decoder = ConstrainedDecoder(model=model, constraint_weight=0.5)
        decoder.add_constraint(Constraint(
            constraint_type="arithmetic",
            pattern="",
        ))
        output = decoder.generate("Compute")
        # At least one violation should reduce confidence
        assert output.confidence < 1.0 or output.constraint_violations == 0

    def test_add_constraint(self):
        decoder = ConstrainedDecoder()
        c = Constraint(constraint_type="arithmetic", pattern="test")
        decoder.add_constraint(c)
        assert c in decoder._constraints

    def test_generate_with_extra_constraints(self):
        decoder = ConstrainedDecoder()
        extra = [Constraint(constraint_type="arithmetic", pattern="")]
        output = decoder.generate("Calculate 3 + 4", constraints=extra)
        assert isinstance(output, ConstrainedOutput)

    def test_default_mock_model_arithmetic(self):
        result = ConstrainedDecoder._default_mock_model("What is 3 + 4?")
        assert "7" in result

    def test_default_mock_model_logic(self):
        result = ConstrainedDecoder._default_mock_model("logic implies reasoning")
        assert "answer" in result.lower()

    def test_default_mock_model_default(self):
        result = ConstrainedDecoder._default_mock_model("something else entirely")
        assert "answer" in result.lower()

    def test_division_constraint(self):
        decoder = ConstrainedDecoder()
        text = "10 / 2 = 3"
        corrected, applied, violations = decoder._apply_arithmetic_constraint(
            text, Constraint(constraint_type="arithmetic", pattern="")
        )
        assert applied
        assert violations == 1
        assert "10 / 2 = 5" in corrected


# ── LogicalLoss tests ──

class TestLogicalLoss:
    def test_zero_loss_correct(self):
        ll = LogicalLoss()
        loss = ll.compute("3 + 4 = 7")
        assert loss == 0.0

    def test_nonzero_loss_incorrect(self):
        ll = LogicalLoss()
        loss = ll.compute("3 + 4 = 8")
        assert loss > 0.0

    def test_arithmetic_loss_correct(self):
        ll = LogicalLoss()
        loss = ll._arithmetic_loss("5 * 6 = 30 and 3 + 4 = 7")
        assert loss == 0.0

    def test_arithmetic_loss_incorrect(self):
        ll = LogicalLoss()
        loss = ll._arithmetic_loss("5 * 6 = 31")
        assert loss > 0.0

    def test_arithmetic_loss_no_pattern(self):
        ll = LogicalLoss()
        loss = ll._arithmetic_loss("No equations here")
        assert loss == 0.0

    def test_consistency_loss_no_contradiction(self):
        ll = LogicalLoss()
        loss = ll._consistency_loss("A is true, B is false")
        assert loss == 0.0

    def test_consistency_loss_truth_contradiction(self):
        ll = LogicalLoss()
        loss = ll._consistency_loss("A is true and A is false")
        assert loss > 0.0

    def test_consistency_loss_assignment_contradiction(self):
        ll = LogicalLoss()
        loss = ll._consistency_loss("x = 5 and later x = 7")
        assert loss > 0.0

    def test_consistency_loss_no_contradiction(self):
        ll = LogicalLoss()
        loss = ll._consistency_loss("x = 5 and y = 7")
        assert loss == 0.0

    def test_compute_with_context(self):
        ll = LogicalLoss()
        loss = ll.compute("3 + 4 = 7", context="arithmetic problem")
        assert loss == 0.0

    def test_custom_weights(self):
        ll = LogicalLoss(arithmetic_weight=1.0, consistency_weight=0.0)
        loss = ll.compute("3 + 4 = 8 and A is true and A is false")
        # Only arithmetic loss should contribute
        assert loss > 0.0

    def test_multiple_arithmetic_errors(self):
        ll = LogicalLoss()
        loss = ll._arithmetic_loss("3 + 4 = 8 and 5 + 6 = 13")
        # Both are wrong: 3+4=7, 5+6=11
        assert loss > 0.0
