"""Tests for DeliberationEngine."""

from __future__ import annotations

import json

import pytest

from src.core.executor import MockLLMClient
from src.core.state import AgentState
from src.meta.registry import ComponentRegistry
from src.meta.prompt_strategy import DefaultPromptStrategy
from src.modification.deliberation import DeliberationEngine, DeliberationResult, RiskAssessment


@pytest.fixture
def engine_with_proposal(mock_llm_with_proposal: MockLLMClient) -> DeliberationEngine:
    return DeliberationEngine(mock_llm_with_proposal)


@pytest.fixture
def engine_defer(mock_llm_defer: MockLLMClient) -> DeliberationEngine:
    return DeliberationEngine(mock_llm_defer)


@pytest.fixture
def state_with_history() -> AgentState:
    return AgentState(
        iteration=10,
        system_prompt="Solve problems.",
        accuracy_history=[0.6, 0.65, 0.7, 0.68, 0.72, 0.71, 0.73, 0.7, 0.69, 0.72],
        modifications_applied=[
            {"target": "prompt_strategy", "success": True},
        ],
    )


@pytest.fixture
def test_registry() -> ComponentRegistry:
    reg = ComponentRegistry()
    reg.register("prompt_strategy", DefaultPromptStrategy())
    return reg


class TestDeliberate:
    def test_propose_modification(
        self, engine_with_proposal: DeliberationEngine, state_with_history: AgentState, test_registry: ComponentRegistry
    ) -> None:
        result = engine_with_proposal.deliberate(
            trigger="performance",
            self_report="Performance stagnating at ~0.7",
            state=state_with_history,
            registry=test_registry,
        )
        assert isinstance(result, DeliberationResult)
        assert result.should_proceed is True
        assert result.proposal is not None
        assert result.proposal.target == "prompt_strategy"
        assert result.trigger == "performance"

    def test_defer_modification(
        self, engine_defer: DeliberationEngine, state_with_history: AgentState, test_registry: ComponentRegistry
    ) -> None:
        result = engine_defer.deliberate(
            trigger="periodic",
            self_report="Performance is fine",
            state=state_with_history,
            registry=test_registry,
        )
        assert result.should_proceed is False
        assert result.proposal is None

    def test_deliberation_with_no_state(self, engine_with_proposal: DeliberationEngine) -> None:
        result = engine_with_proposal.deliberate(trigger="performance")
        assert isinstance(result, DeliberationResult)
        # Should still work without state/registry
        assert result.reasoning != ""


class TestRiskAssessment:
    def test_low_risk_assessment(self, engine_with_proposal: DeliberationEngine, state_with_history: AgentState, test_registry: ComponentRegistry) -> None:
        result = engine_with_proposal.deliberate(
            trigger="performance",
            state=state_with_history,
            registry=test_registry,
        )
        assert result.risk_assessment.level in ("low", "medium")

    def test_high_risk_blocks_proceed(self) -> None:
        """Critical risk should block proceeding."""
        llm = MockLLMClient(default_response=json.dumps({
            "action": "modify",
            "target": "prompt_strategy",
            "method_name": "prepare_prompt",
            "description": "Risky change",
            "code": "def prepare_prompt(self, task, examples):\n    return 'x' * 1000\n",
            "risk": "high",
            "rationale": "Worth trying",
        }))
        engine = DeliberationEngine(llm)

        # State with many recent failures to push risk to critical
        state = AgentState(
            iteration=10,
            modifications_applied=[
                {"target": "prompt_strategy", "success": False},
                {"target": "prompt_strategy", "success": False},
                {"target": "prompt_strategy", "success": False},
            ],
        )
        result = engine.deliberate(trigger="performance", state=state)
        # With high self-assessed risk + recent failures, should be critical
        assert result.risk_assessment.level in ("high", "critical")

    def test_empty_code_does_not_proceed(self) -> None:
        llm = MockLLMClient(default_response=json.dumps({
            "action": "modify",
            "target": "prompt_strategy",
            "code": "",
            "risk": "low",
        }))
        engine = DeliberationEngine(llm)
        result = engine.deliberate(trigger="performance")
        assert result.should_proceed is False


class TestDeliberationResult:
    def test_to_dict(self) -> None:
        result = DeliberationResult(
            should_proceed=True,
            reasoning="Test",
            trigger="performance",
            risk_assessment=RiskAssessment(level="low", score=0.1),
        )
        d = result.to_dict()
        assert d["should_proceed"] is True
        assert d["trigger"] == "performance"
        assert d["risk_assessment"]["level"] == "low"


class TestParseProposal:
    def test_parse_valid_json(self) -> None:
        engine = DeliberationEngine(MockLLMClient())
        response = json.dumps({
            "action": "modify",
            "target": "prompt_strategy",
            "method_name": "prepare_prompt",
            "code": "def prepare_prompt(self, task, ex): return 'test'",
            "risk": "low",
        })
        proposal = engine._parse_proposal(response)
        assert proposal is not None
        assert proposal.target == "prompt_strategy"

    def test_parse_defer_returns_none(self) -> None:
        engine = DeliberationEngine(MockLLMClient())
        response = json.dumps({"action": "defer"})
        proposal = engine._parse_proposal(response)
        assert proposal is None

    def test_parse_invalid_json_returns_none(self) -> None:
        engine = DeliberationEngine(MockLLMClient())
        proposal = engine._parse_proposal("not json at all")
        assert proposal is None

    def test_parse_json_embedded_in_text(self) -> None:
        engine = DeliberationEngine(MockLLMClient())
        response = 'Here is my proposal: {"action": "modify", "target": "prompt_strategy", "code": "def f(): pass", "risk": "low"} end'
        proposal = engine._parse_proposal(response)
        assert proposal is not None
        assert proposal.target == "prompt_strategy"
