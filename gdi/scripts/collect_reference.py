#!/usr/bin/env python3
"""Collect reference outputs for GDI baseline."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reference.collector import ReferenceCollector
from src.reference.store import ReferenceStore


def main():
    """Collect reference outputs from a mock agent."""
    store = ReferenceStore(
        os.path.join(os.path.dirname(__file__), "..", "data", "reference", "baseline.json")
    )

    probe_tasks = [
        "Explain the concept of recursion in programming.",
        "What are the benefits of test-driven development?",
        "Describe how a hash table works internally.",
    ]

    # Mock agent for demonstration
    class MockAgent:
        def run(self, task: str) -> str:
            return f"Response to: {task}"

    collector = ReferenceCollector(samples_per_task=3)
    outputs = collector.collect(MockAgent(), probe_tasks)

    store.save({
        "outputs": outputs.outputs,
        "task_outputs": outputs.task_outputs,
        "metadata": outputs.metadata,
    })

    print(f"Collected {len(outputs.outputs)} reference outputs.")
    print(f"Saved to {store.store_path}")


if __name__ == "__main__":
    main()
