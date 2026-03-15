"""Deliberation engine for meta-reasoning about self-modification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.modification.modifier import ModificationProposal

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Assessment of modification risk."""

    level: str = "low"  # low, medium, high, critical
    factors: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "factors": self.factors, "score": self.score}


@dataclass
class DeliberationResult:
    """Result of a deliberation process."""

    should_proceed: bool = False
    proposal: ModificationProposal | None = None
    risk_assessment: RiskAssessment = field(default_factory=RiskAssessment)
    reasoning: str = ""
    trigger: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_proceed": self.should_proceed,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "risk_assessment": self.risk_assessment.to_dict(),
            "reasoning": self.reasoning,
            "trigger": self.trigger,
        }


class DeliberationEngine:
    """Uses LLM for meta-reasoning about whether and how to self-modify."""

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        self.llm = llm_client
        self.config = config or {}
        self.deliberation_depth = self.config.get("deliberation_depth", 2)

    def deliberate(
        self,
        trigger: str = "performance",
        self_report: str = "",
        state: Any = None,
        registry: Any = None,
    ) -> DeliberationResult:
        """Run the deliberation process to decide on self-modification."""
        result = DeliberationResult(trigger=trigger)

        # Build prompt
        prompt = self._build_deliberation_prompt(trigger, self_report, state, registry)

        # Query LLM
        try:
            response = self.llm.generate(
                prompt,
                system_prompt="You are a meta-reasoning engine. Analyze the agent's performance and propose modifications if needed. Respond in JSON.",
            )
            result.reasoning = response

            # Parse proposal
            proposal = self._parse_proposal(response)
            if proposal:
                risk = self._evaluate_risk(proposal, state)
                result.risk_assessment = risk
                result.proposal = proposal
                result.should_proceed = self._should_proceed(proposal, risk)
            else:
                result.should_proceed = False

        except Exception as e:
            logger.error(f"Deliberation failed: {e}")
            result.reasoning = f"Error: {e}"
            result.should_proceed = False

        return result

    def _build_deliberation_prompt(
        self,
        trigger: str,
        self_report: str,
        state: Any = None,
        registry: Any = None,
    ) -> str:
        """Build the deliberation prompt."""
        parts: list[str] = [
            "=== Self-Modification Deliberation ===",
            f"\nTrigger: {trigger}",
        ]

        if self_report:
            parts.append(f"\nAgent Self-Report:\n{self_report}")

        if state:
            parts.append(f"\nCurrent iteration: {getattr(state, 'iteration', 'unknown')}")
            acc_history = getattr(state, "accuracy_history", [])
            if acc_history:
                parts.append(f"Recent accuracy: {acc_history[-5:]}")
            mods = getattr(state, "modifications_applied", [])
            parts.append(f"Modifications so far: {len(mods)}")

        if registry:
            components = registry.list_components()
            parts.append(f"\nModifiable components: {components}")

        parts.extend([
            "\nYou can propose a modification to one of the modifiable components.",
            "Respond with a JSON object containing:",
            '  "action": "modify" or "defer"',
            '  "target": component name',
            '  "method_name": method to modify',
            '  "description": what the change does',
            '  "code": new Python function code',
            '  "risk": "low", "medium", or "high"',
            '  "rationale": why this change will help',
        ])

        return "\n".join(parts)

    def _parse_proposal(self, response: str) -> ModificationProposal | None:
        """Parse a modification proposal from LLM response."""
        try:
            # Try to find JSON in the response
            text = response.strip()

            # Look for JSON block
            if "{" in text:
                start = text.index("{")
                # Find matching closing brace
                depth = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break

                json_str = text[start:end]
                data = json.loads(json_str)

                action = data.get("action", "defer")
                if action == "defer":
                    return None

                return ModificationProposal(
                    target=data.get("target", ""),
                    description=data.get("description", ""),
                    code=data.get("code", ""),
                    risk=data.get("risk", "low"),
                    rationale=data.get("rationale", ""),
                    method_name=data.get("method_name", ""),
                )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse proposal: {e}")

        return None

    def _evaluate_risk(
        self, proposal: ModificationProposal, state: Any = None
    ) -> RiskAssessment:
        """Evaluate the risk of a proposed modification."""
        factors: list[str] = []
        score = 0.0

        # Code length risk
        if proposal.code and len(proposal.code) > 500:
            factors.append("Large code change")
            score += 0.3

        # Target risk
        if proposal.target in ("prompt_strategy",):
            score += 0.1
        elif proposal.target in ("reasoning_strategy",):
            score += 0.2
            factors.append("Modifying reasoning strategy")

        # Risk label
        if proposal.risk == "high":
            score += 0.3
            factors.append("Self-assessed high risk")
        elif proposal.risk == "medium":
            score += 0.15

        # Recent failures
        if state:
            mods = getattr(state, "modifications_applied", [])
            recent_failures = sum(1 for m in mods[-3:] if not m.get("success", True))
            if recent_failures > 0:
                score += 0.2 * recent_failures
                factors.append(f"{recent_failures} recent failed modifications")

        level = "low"
        if score > 0.7:
            level = "critical"
        elif score > 0.5:
            level = "high"
        elif score > 0.25:
            level = "medium"

        return RiskAssessment(level=level, factors=factors, score=score)

    def _should_proceed(
        self, proposal: ModificationProposal, risk: RiskAssessment
    ) -> bool:
        """Decide whether to proceed with the modification."""
        if risk.level == "critical":
            return False
        if not proposal.code:
            return False
        if not proposal.target:
            return False
        return True
