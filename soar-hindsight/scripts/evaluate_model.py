#!/usr/bin/env python3
"""Evaluate a fine-tuned model against base."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finetuning.evaluation import Evaluator


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "soar-ft-v1"

    evaluator = Evaluator()

    print("Evaluating base model...")
    base_metrics = evaluator.evaluate("base", is_base=True)
    print(json.dumps(base_metrics, indent=2))

    print(f"\nEvaluating {model_name}...")
    ft_metrics = evaluator.evaluate(model_name, is_base=False)
    print(json.dumps(ft_metrics, indent=2))

    print("\nComparison:")
    comparison = evaluator.compare("base", model_name)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
