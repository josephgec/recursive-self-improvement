#!/usr/bin/env python3
"""Run a simple REPL server for interactive testing."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends.local import LocalREPL
from src.safety.policy import SafetyPolicy


def main():
    """Run an interactive REPL server."""
    policy = SafetyPolicy()
    repl = LocalREPL(policy=policy)

    print("RLM-REPL Interactive Server")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 40)

    while True:
        try:
            code = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break

        if code.strip() in ("exit", "quit"):
            break

        if not code.strip():
            continue

        try:
            result = repl.execute(code)
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            if result.error:
                print(f"Error: {result.error}")
        except Exception as e:
            print(f"REPL Error: {e}")

    repl.shutdown()
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
