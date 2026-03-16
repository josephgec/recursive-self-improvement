"""RLM Wrapper: wraps pipeline operations with long-context reasoning."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.pipeline.state import PipelineState
from src.outer_loop.strategy_evolver import Candidate
from src.scaling.context_manager import ContextManager
from src.scaling.memory_bridge import MemoryBridge


class RLMWrapper:
    """Wraps pipeline operations with RLM (Reasoning Language Model) context.

    Provides evolve, verify, and inspect operations within an RLM session
    that has access to the full codebase, history, and dataset.
    """

    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        memory_bridge: Optional[MemoryBridge] = None,
        rlm_engine: Optional[Callable] = None,
    ):
        self._context = context_manager or ContextManager()
        self._memory = memory_bridge or MemoryBridge()
        self._rlm = rlm_engine or self._mock_rlm
        self._session_active = False

    def evolve_with_context(self, state: PipelineState, n: int = 5) -> List[Candidate]:
        """Generate candidates using RLM with full context."""
        context = self._prepare_context(state)
        result = self._rlm("evolve", context, n=n)
        candidates = []
        for i, proposal in enumerate(result.get("proposals", [])):
            candidates.append(Candidate(
                target=proposal.get("target", "default"),
                proposed_code=proposal.get("code", state.agent_code.code + f"\n# RLM evolved {i}"),
                description=proposal.get("description", f"RLM evolution {i}"),
                operator="rlm_evolve",
            ))
        return candidates

    def verify_with_context(self, candidate: Candidate, state: PipelineState) -> Dict[str, Any]:
        """Verify a candidate using RLM with full context."""
        context = self._prepare_context(state)
        context["candidate"] = candidate.to_dict()
        result = self._rlm("verify", context)
        return {
            "verified": result.get("verified", True),
            "confidence": result.get("confidence", 0.8),
            "reasoning": result.get("reasoning", "RLM verification passed"),
        }

    def inspect_with_context(self, state: PipelineState) -> Dict[str, Any]:
        """Inspect the current state using RLM."""
        context = self._prepare_context(state)
        result = self._rlm("inspect", context)
        return {
            "assessment": result.get("assessment", "stable"),
            "suggestions": result.get("suggestions", []),
            "risk_level": result.get("risk_level", "low"),
        }

    def wrap_iteration(self, state: PipelineState, iteration_fn: Callable) -> Any:
        """Wrap a pipeline iteration with RLM context."""
        self._session_active = True
        self._memory.save_state_to_repl(state)
        try:
            result = iteration_fn(state)
            return result
        finally:
            self._session_active = False

    @property
    def session_active(self) -> bool:
        return self._session_active

    def _prepare_context(self, state: PipelineState) -> Dict[str, Any]:
        """Prepare context for RLM operations."""
        self._context.load_codebase(state.agent_code.code)
        history = [p.to_dict() for p in state.performance_history]
        self._context.load_history(history)
        return self._context.get_context()

    @staticmethod
    def _mock_rlm(operation: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Mock RLM engine for testing."""
        if operation == "evolve":
            n = kwargs.get("n", 3)
            proposals = []
            for i in range(n):
                proposals.append({
                    "target": "strategy_evolver",
                    "code": f"# RLM proposal {i}\npass",
                    "description": f"RLM-generated proposal {i}",
                })
            return {"proposals": proposals}
        elif operation == "verify":
            return {"verified": True, "confidence": 0.85, "reasoning": "Mock verification OK"}
        elif operation == "inspect":
            return {
                "assessment": "stable",
                "suggestions": ["Consider optimizing candidate generation"],
                "risk_level": "low",
            }
        return {}
