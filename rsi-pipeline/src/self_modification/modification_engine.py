"""Modification engine: applies verified candidates to agent code."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional

from src.pipeline.state import PipelineState, AgentCodeSnapshot
from src.verification.dual_verifier import VerifiedCandidate
from src.self_modification.target_registry import TargetRegistry
from src.self_modification.rollback_bridge import RollbackBridge
from src.self_modification.audit_bridge import AuditBridge
from src.self_modification.safety_gate import SafetyGate


@dataclass
class ModificationResult:
    """Result of applying a modification."""
    applied: bool = False
    reason: str = ""
    candidate_id: str = ""
    target: str = ""
    old_version: int = 0
    new_version: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModificationEngine:
    """Applies verified candidates to the agent's code with safety preconditions."""

    def __init__(
        self,
        target_registry: Optional[TargetRegistry] = None,
        rollback_bridge: Optional[RollbackBridge] = None,
        audit_bridge: Optional[AuditBridge] = None,
        safety_gate: Optional[SafetyGate] = None,
        cooldown_iterations: int = 2,
        complexity_ceiling: int = 1000,
        validator: Optional[Callable] = None,
    ):
        self._target_registry = target_registry or TargetRegistry()
        self._rollback = rollback_bridge or RollbackBridge()
        self._audit = audit_bridge or AuditBridge()
        self._safety_gate = safety_gate or SafetyGate()
        self._cooldown = cooldown_iterations
        self._complexity_ceiling = complexity_ceiling
        self._validator = validator
        self._last_modification_iteration: int = -999

    def apply(self, verified: VerifiedCandidate, state: PipelineState) -> ModificationResult:
        """Apply a verified candidate modification to the agent state.

        Checks preconditions:
        1. Target must be allowed (whitelist)
        2. Cooldown must have elapsed
        3. Code complexity must be under ceiling
        4. Pre-modification safety gate must pass
        """
        candidate = verified.candidate
        result = ModificationResult(
            candidate_id=candidate.candidate_id,
            target=candidate.target,
            old_version=state.agent_code.version,
        )

        # Check 1: Target whitelist
        if not self._target_registry.is_allowed(candidate.target):
            result.reason = f"target_not_allowed: {candidate.target}"
            if self._target_registry.is_forbidden(candidate.target):
                result.reason = f"target_forbidden: {candidate.target}"
            return result

        # Check 2: Cooldown
        iterations_since = state.iteration - self._last_modification_iteration
        if iterations_since < self._cooldown:
            result.reason = f"cooldown_active: {iterations_since}/{self._cooldown}"
            return result

        # Check 3: Complexity ceiling
        code_complexity = len(candidate.proposed_code)
        if code_complexity > self._complexity_ceiling:
            result.reason = f"complexity_exceeded: {code_complexity}/{self._complexity_ceiling}"
            return result

        # Check 4: Safety gate
        safety_check = self._safety_gate.check_pre_modification(candidate, state)
        if not safety_check["allowed"]:
            result.reason = f"safety_gate: {'; '.join(safety_check['reasons'])}"
            return result

        # Create checkpoint before modification
        self._rollback.checkpoint(state)

        # Apply modification
        old_code = state.agent_code.code
        state.agent_code = AgentCodeSnapshot(
            code=candidate.proposed_code,
            version=state.agent_code.version + 1,
            target=candidate.target,
        )

        # Validate if validator provided
        if self._validator:
            try:
                valid = self._validator(state)
                if not valid:
                    # Auto-rollback on validation failure
                    self._rollback.rollback(state)
                    self._audit.log_rollback("validation_failed", state.iteration)
                    result.reason = "validation_failed"
                    return result
            except Exception as e:
                self._rollback.rollback(state)
                self._audit.log_rollback(f"validation_error: {e}", state.iteration)
                result.reason = f"validation_error: {e}"
                return result

        # Success
        result.applied = True
        result.new_version = state.agent_code.version
        self._last_modification_iteration = state.iteration

        # Audit log
        self._audit.log_modification(
            candidate_id=candidate.candidate_id,
            target=candidate.target,
            old_code=old_code,
            new_code=candidate.proposed_code,
        )

        # Record in state history
        state.modification_history.append({
            "iteration": state.iteration,
            "candidate_id": candidate.candidate_id,
            "target": candidate.target,
            "version": state.agent_code.version,
        })

        return result

    def rollback(self, state: PipelineState) -> bool:
        """Rollback the most recent modification."""
        success = self._rollback.rollback(state)
        if success:
            self._audit.log_rollback("manual_rollback", state.iteration)
        return success

    @property
    def audit(self) -> AuditBridge:
        return self._audit

    @property
    def rollback_bridge(self) -> RollbackBridge:
        return self._rollback
