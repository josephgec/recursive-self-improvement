#!/usr/bin/env python3
"""Dry-run: validate pipeline setup without modifying state."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.state import PipelineState, AgentCodeSnapshot
from src.pipeline.config import PipelineConfig
from src.outer_loop.strategy_evolver import StrategyEvolver
from src.verification.dual_verifier import DualVerifier
from src.safety.gdi_monitor import GDIMonitor
from src.safety.constraint_enforcer import ConstraintEnforcer


def main():
    config = PipelineConfig()
    print("Pipeline configuration loaded successfully.")

    initial_code = "def solve(x): return x + 1"
    state = PipelineState(
        agent_code=AgentCodeSnapshot(code=initial_code),
        original_code=AgentCodeSnapshot(code=initial_code),
    )
    state.performance.accuracy = 0.7

    # Test candidate generation
    evolver = StrategyEvolver()
    candidates = evolver.generate_candidates(state, 3)
    print(f"Generated {len(candidates)} candidates.")

    # Test verification
    verifier = DualVerifier()
    verified = verifier.verify_all(candidates, state)
    print(f"Verified {len(verified)} candidates.")

    # Test safety checks
    gdi = GDIMonitor()
    score = gdi.compute(state.agent_code.code, state.original_code.code)
    print(f"GDI score: {score:.4f}")

    enforcer = ConstraintEnforcer()
    verdict = enforcer.check_all(state)
    print(f"Constraints satisfied: {verdict.satisfied}")

    print("\nDry run complete. All components functional.")


if __name__ == "__main__":
    main()
