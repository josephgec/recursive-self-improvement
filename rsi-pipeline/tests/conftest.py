"""Shared fixtures and mocks for RSI pipeline tests."""
from __future__ import annotations

import sys
import os
import copy
from typing import Any, Dict, List, Optional

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.state import PipelineState, AgentCodeSnapshot, PerformanceRecord, SafetyStatus
from src.pipeline.config import PipelineConfig
from src.pipeline.iteration import IterationResult
from src.outer_loop.strategy_evolver import Candidate, StrategyEvolver
from src.verification.dual_verifier import DualVerifier, VerifiedCandidate
from src.verification.empirical_gate import EmpiricalGate, EmpiricalResult
from src.verification.compactness_gate import CompactnessGate, CompactnessResult
from src.self_modification.modification_engine import ModificationEngine
from src.safety.gdi_monitor import GDIMonitor
from src.safety.constraint_enforcer import ConstraintEnforcer
from src.safety.car_tracker import CARTracker
from src.safety.emergency_stop import EmergencyStop
from src.outer_loop.hindsight_adapter import HindsightAdapter
from src.tracking.iteration_logger import IterationLogger
from src.tracking.improvement_curve import ImprovementCurveTracker


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns predictable code mutations."""

    def __init__(self, improvement_schedule: Optional[List[bool]] = None):
        self.call_count = 0
        self.improvement_schedule = improvement_schedule or [True, True, True, True, True]

    def __call__(self, current_code: str, target: str, operator: str) -> str:
        """Generate a mock code mutation."""
        idx = self.call_count % len(self.improvement_schedule)
        self.call_count += 1
        if self.improvement_schedule[idx]:
            return current_code + f"\n# improved_{operator}_{target}_{self.call_count}"
        else:
            return current_code + f"\n# no_change_{self.call_count}"


# ---------------------------------------------------------------------------
# Mock Agent
# ---------------------------------------------------------------------------

class MockAgent:
    """Mock agent with modifiable code, inspection, and modification capabilities."""

    def __init__(self, code: str = "def solve(x): return x + 1", accuracy: float = 0.7):
        self.code = code
        self.accuracy = accuracy
        self._history: List[str] = [code]

    def inspect(self) -> Dict[str, Any]:
        """Inspect agent state."""
        return {
            "code": self.code,
            "accuracy": self.accuracy,
            "version": len(self._history) - 1,
        }

    def modify(self, new_code: str) -> bool:
        """Modify agent code."""
        self._history.append(new_code)
        self.code = new_code
        return True

    def rollback(self) -> bool:
        """Rollback to previous code."""
        if len(self._history) < 2:
            return False
        self._history.pop()
        self.code = self._history[-1]
        return True

    @property
    def history(self) -> List[str]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def sample_code():
    return "def solve(x): return x + 1"


@pytest.fixture
def sample_state(sample_code):
    """Create a sample pipeline state."""
    state = PipelineState(
        agent_code=AgentCodeSnapshot(code=sample_code, version=0),
        original_code=AgentCodeSnapshot(code=sample_code, version=0),
        performance=PerformanceRecord(accuracy=0.7, entropy=1.0),
    )
    return state


@pytest.fixture
def sample_candidate(sample_code):
    """Create a sample candidate."""
    return Candidate(
        candidate_id="test_candidate_001",
        target="strategy_evolver",
        proposed_code=sample_code + "\n# improved_mutate",
        description="Test mutation",
        operator="mutate",
    )


@pytest.fixture
def sample_verified_candidate(sample_candidate):
    """Create a sample verified candidate."""
    return VerifiedCandidate(
        candidate=sample_candidate,
        empirical=EmpiricalResult(
            candidate_id=sample_candidate.candidate_id,
            passed=True,
            pass_rate=0.9,
            accuracy=0.85,
        ),
        compactness=CompactnessResult(
            candidate_id=sample_candidate.candidate_id,
            passed=True,
            bdm_score=50.0,
            complexity_ratio=0.1,
            code_length=100,
        ),
        combined_score=0.75,
    )


@pytest.fixture
def default_config():
    return PipelineConfig()


@pytest.fixture
def strategy_evolver(mock_llm):
    return StrategyEvolver(llm=mock_llm)


@pytest.fixture
def dual_verifier():
    return DualVerifier()


@pytest.fixture
def modification_engine():
    return ModificationEngine(cooldown_iterations=0)


@pytest.fixture
def gdi_monitor():
    return GDIMonitor(threshold=0.3)


@pytest.fixture
def constraint_enforcer():
    return ConstraintEnforcer(accuracy_floor=0.6, entropy_floor=0.1, drift_ceiling=0.5)


@pytest.fixture
def car_tracker():
    return CARTracker(min_ratio=0.5)


@pytest.fixture
def emergency_stop():
    return EmergencyStop(car_threshold=0.5, max_consecutive_rollbacks=3)


@pytest.fixture
def hindsight_adapter():
    return HindsightAdapter()


@pytest.fixture
def iteration_logger():
    return IterationLogger()


@pytest.fixture
def improvement_tracker():
    return ImprovementCurveTracker()


def make_evaluator(accuracy: float):
    """Create an evaluator that returns a fixed accuracy."""
    def evaluator(state):
        state.performance.accuracy = accuracy
        return accuracy
    return evaluator


def make_failing_evaluator():
    """Create an evaluator that lowers accuracy."""
    def evaluator(state):
        new_acc = state.performance.accuracy * 0.5
        state.performance.accuracy = new_acc
        return new_acc
    return evaluator
