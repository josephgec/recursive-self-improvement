"""Tests for CodeModifier."""

from __future__ import annotations

import pytest

from src.modification.modifier import CodeModifier, ModificationProposal, ModificationResult
from src.meta.prompt_strategy import DefaultPromptStrategy
from src.meta.registry import ComponentRegistry


@pytest.fixture
def setup_registry() -> tuple[ComponentRegistry, DefaultPromptStrategy]:
    reg = ComponentRegistry()
    strategy = DefaultPromptStrategy()
    reg.register("prompt_strategy", strategy)
    return reg, strategy


class TestValidateProposal:
    def test_valid_proposal(self, modifier: CodeModifier, valid_proposal: ModificationProposal) -> None:
        result = modifier.validate_proposal(valid_proposal)
        assert result["valid"] is True

    def test_forbidden_target(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="validation.suite",
            code="def x(): pass",
        )
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is False
        assert "forbidden" in result["reason"].lower()

    def test_not_allowed_target(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="some_random_target",
            code="def x(): pass",
        )
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is False
        assert "not in allowed" in result["reason"].lower()

    def test_syntax_error_rejected(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="prompt_strategy",
            code="def broken(:\n    pass",
        )
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is False
        assert "syntax" in result["reason"].lower()

    def test_forbidden_import_rejected(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="prompt_strategy",
            code="import os\ndef f(): pass\n",
        )
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is False
        assert "import" in result["reason"].lower()

    def test_forbidden_from_import_rejected(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="prompt_strategy",
            code="from subprocess import call\ndef f(): pass\n",
        )
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is False

    def test_safe_import_allowed(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="prompt_strategy",
            code="import math\ndef f(): return math.sqrt(4)\n",
        )
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is True

    def test_high_risk_warning(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="prompt_strategy",
            code="def f(): pass",
            risk="high",
        )
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is True
        assert len(result["warnings"]) > 0


class TestApplyModification:
    def test_apply_valid_modification(
        self, modifier: CodeModifier, setup_registry: tuple
    ) -> None:
        registry, strategy = setup_registry
        proposal = ModificationProposal(
            target="prompt_strategy",
            method_name="choose_reasoning_mode",
            code="def choose_reasoning_mode(self, task, recent_results):\n    return 'direct'\n",
            risk="low",
        )
        result = modifier.apply_modification(proposal, registry)
        assert result.success is True
        # Verify the method was patched
        mode = strategy.choose_reasoning_mode(None, [])
        assert mode == "direct"

    def test_apply_without_registry_fails(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="prompt_strategy",
            method_name="choose_reasoning_mode",
            code="def choose_reasoning_mode(self, task, r): return 'direct'\n",
        )
        result = modifier.apply_modification(proposal, registry=None)
        assert result.success is False
        assert "registry" in result.error.lower()


class TestMonkeyPatch:
    def test_monkey_patch_replaces_method(self, modifier: CodeModifier) -> None:
        strategy = DefaultPromptStrategy()
        assert strategy.choose_reasoning_mode(None, []) == "cot"

        code = "def choose_reasoning_mode(self, task, recent_results):\n    return 'code'\n"
        result = modifier.monkey_patch(strategy, "choose_reasoning_mode", code)
        assert result is not None
        assert strategy.choose_reasoning_mode(None, []) == "code"

    def test_monkey_patch_with_bad_code_returns_none(self, modifier: CodeModifier) -> None:
        strategy = DefaultPromptStrategy()
        result = modifier.monkey_patch(strategy, "some_method", "this is not valid python!!!")
        assert result is None


class TestRevert:
    def test_revert_restores_original(
        self, modifier: CodeModifier, setup_registry: tuple
    ) -> None:
        registry, strategy = setup_registry
        original_mode = strategy.choose_reasoning_mode(None, [])
        assert original_mode == "cot"

        proposal = ModificationProposal(
            target="prompt_strategy",
            method_name="choose_reasoning_mode",
            code="def choose_reasoning_mode(self, task, recent_results):\n    return 'direct'\n",
        )
        result = modifier.apply_modification(proposal, registry)
        assert result.success
        assert strategy.choose_reasoning_mode(None, []) == "direct"

        reverted = modifier.revert(result)
        assert reverted is True
        assert strategy.choose_reasoning_mode(None, []) == "cot"


class TestForbiddenTargets:
    def test_audit_logger_cannot_be_modified(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="audit.logger",
            code="def log(): pass",
        )
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is False

    def test_rollback_mechanism_cannot_be_modified(self, modifier: CodeModifier) -> None:
        proposal = ModificationProposal(
            target="rollback.mechanism",
            code="def rollback(): pass",
        )
        # Not in allowed_targets, would fail on that check too
        result = modifier.validate_proposal(proposal)
        assert result["valid"] is False
