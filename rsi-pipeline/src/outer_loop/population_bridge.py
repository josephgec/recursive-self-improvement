"""Population bridge: syncs state with SOAR population."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.pipeline.state import PipelineState
from src.outer_loop.strategy_evolver import Candidate


class PopulationBridge:
    """Bridges the pipeline state with a SOAR-style population."""

    def __init__(self, population_size: int = 10):
        self._population: List[Dict[str, Any]] = []
        self._population_size = population_size

    def sync(self, state: PipelineState, candidates: Optional[List[Candidate]] = None) -> None:
        """Sync the population with current pipeline state."""
        entry = {
            "iteration": state.iteration,
            "accuracy": state.performance.accuracy,
            "code_version": state.agent_code.version,
            "state_id": state.state_id,
        }
        self._population.append(entry)
        # Keep population within size limit
        if len(self._population) > self._population_size:
            # Remove lowest accuracy entries
            self._population.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
            self._population = self._population[:self._population_size]

    @property
    def population(self) -> List[Dict[str, Any]]:
        return list(self._population)

    @property
    def size(self) -> int:
        return len(self._population)

    def get_best(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get top-n population members by accuracy."""
        sorted_pop = sorted(self._population, key=lambda x: x.get("accuracy", 0), reverse=True)
        return sorted_pop[:n]

    def clear(self) -> None:
        """Clear the population."""
        self._population.clear()
