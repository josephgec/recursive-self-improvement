#!/usr/bin/env python3
"""Run a single RLM query from the command line."""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.completion import completion
from src.core.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single RLM query")
    parser.add_argument("prompt", help="The query to run")
    parser.add_argument("--context-file", help="Path to context file")
    parser.add_argument("--context", default="", help="Inline context string")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    ctx = args.context
    if args.context_file:
        with open(args.context_file) as f:
            ctx = f.read()

    # Use a simple mock LLM for demonstration
    from tests.conftest import MockLLM
    llm = MockLLM()

    result = completion(prompt=args.prompt, context=ctx, llm=llm, config=config)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
