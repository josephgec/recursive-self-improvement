#!/usr/bin/env python3
"""Run the full RSI pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.state import PipelineState, AgentCodeSnapshot
from src.pipeline.config import PipelineConfig
from src.pipeline.orchestrator import RSIPipelineOrchestrator
from src.outer_loop.strategy_evolver import StrategyEvolver
from src.verification.dual_verifier import DualVerifier
from src.self_modification.modification_engine import ModificationEngine
from src.safety.gdi_monitor import GDIMonitor
from src.safety.constraint_enforcer import ConstraintEnforcer
from src.safety.car_tracker import CARTracker
from src.safety.emergency_stop import EmergencyStop
from src.outer_loop.hindsight_adapter import HindsightAdapter


def main():
    config = PipelineConfig()

    initial_code = "def solve(x): return x + 1"
    state = PipelineState(
        agent_code=AgentCodeSnapshot(code=initial_code),
        original_code=AgentCodeSnapshot(code=initial_code),
    )
    state.performance.accuracy = 0.7

    orchestrator = RSIPipelineOrchestrator(
        config=config,
        strategy_evolver=StrategyEvolver(),
        dual_verifier=DualVerifier(),
        modification_engine=ModificationEngine(),
        gdi_monitor=GDIMonitor(),
        constraint_enforcer=ConstraintEnforcer(),
        car_tracker=CARTracker(),
        emergency_stop=EmergencyStop(),
        hindsight_adapter=HindsightAdapter(),
    )

    result = orchestrator.run(state, max_iterations=10)
    print(f"Pipeline completed: {result.total_iterations} iterations, "
          f"{result.successful_improvements} improvements, "
          f"final accuracy: {result.final_accuracy:.4f}")


if __name__ == "__main__":
    main()
