#!/usr/bin/env python3
"""Check current budget status across all levels."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cost.budget_manager import BudgetManager


def main():
    budget = BudgetManager()
    print("Budget Status:")
    for state in budget.get_all_states():
        print(f"  {state.level}: ${state.remaining:.2f} remaining "
              f"(${state.limit:.2f} limit, {state.utilization:.0%} used)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
