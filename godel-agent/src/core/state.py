"""Agent state management with serialization and lineage tracking."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import dill


@dataclass
class AgentState:
    """Complete snapshot of agent state at a point in time."""

    iteration: int = 0
    prompt_template: str = ""
    system_prompt: str = ""
    few_shot_examples: list[dict[str, Any]] = field(default_factory=list)
    few_shot_selector_code: str = ""
    reasoning_strategy_code: str = ""
    accuracy_history: list[float] = field(default_factory=list)
    modifications_applied: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    state_id: str = ""
    parent_state_id: str = ""

    def __post_init__(self) -> None:
        if not self.state_id:
            self.state_id = self._compute_id()

    def _compute_id(self) -> str:
        content = json.dumps(
            {
                "iteration": self.iteration,
                "prompt_template": self.prompt_template,
                "system_prompt": self.system_prompt,
                "few_shot_selector_code": self.few_shot_selector_code,
                "reasoning_strategy_code": self.reasoning_strategy_code,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateManager:
    """Manages agent state capture, restore, and persistence."""

    def __init__(self, checkpoint_dir: str | Path = "data/checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[AgentState] = []

    def capture(self, state: AgentState) -> AgentState:
        """Record a state snapshot."""
        if self._history:
            state.parent_state_id = self._history[-1].state_id
        state.state_id = state._compute_id()
        self._history.append(state)
        return state

    def restore(self, state_id: str) -> AgentState | None:
        """Restore a previously captured state by ID."""
        for s in self._history:
            if s.state_id == state_id:
                return s
        return None

    def create_initial_state(self, system_prompt: str = "", **kwargs: Any) -> AgentState:
        """Create and capture an initial agent state."""
        state = AgentState(
            iteration=0,
            system_prompt=system_prompt,
            timestamp=time.time(),
            **kwargs,
        )
        return self.capture(state)

    def save_to_disk(self, state: AgentState, label: str = "") -> Path:
        """Serialize state to disk using dill."""
        filename = f"state_{state.state_id}"
        if label:
            filename += f"_{label}"
        filename += ".pkl"
        path = self.checkpoint_dir / filename
        with open(path, "wb") as f:
            dill.dump(state, f)
        return path

    def load_from_disk(self, path: str | Path) -> AgentState:
        """Load state from disk."""
        with open(path, "rb") as f:
            state = dill.load(f)
        return state

    def get_lineage(self, state_id: str) -> list[AgentState]:
        """Get the chain of states leading to the given state_id."""
        id_to_state = {s.state_id: s for s in self._history}
        lineage: list[AgentState] = []
        current_id = state_id
        while current_id and current_id in id_to_state:
            state = id_to_state[current_id]
            lineage.append(state)
            current_id = state.parent_state_id
        lineage.reverse()
        return lineage

    def diff_states(self, state_a: AgentState, state_b: AgentState) -> dict[str, Any]:
        """Compare two states and return the differences."""
        diff: dict[str, Any] = {}
        a_dict = state_a.to_dict()
        b_dict = state_b.to_dict()
        for key in a_dict:
            if a_dict[key] != b_dict.get(key):
                diff[key] = {"before": a_dict[key], "after": b_dict.get(key)}
        return diff

    @property
    def history(self) -> list[AgentState]:
        return list(self._history)

    @property
    def latest(self) -> AgentState | None:
        return self._history[-1] if self._history else None
