#!/usr/bin/env python3
"""Run a modification through the staging environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.self_mod.staging_env import StagingEnvironment


def main():
    staging = StagingEnvironment()

    agent_state = {"quality": 0.85, "modifier": 0.0, "version": "1.0"}
    candidate = {
        "id": "candidate_001",
        "changes": {"modifier": 0.02, "version": "1.1"},
    }

    result = staging.test_modification(agent_state, candidate)
    print(f"Staging result: {'PASS' if result.passed else 'FAIL'}")
    print(f"  Original score: {result.original_score:.3f}")
    print(f"  Modified score: {result.modified_score:.3f}")
    print(f"  Improvement: {result.improvement:.3f}")
    print(f"  Detail: {result.detail}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
