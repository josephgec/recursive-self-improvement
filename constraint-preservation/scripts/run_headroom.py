#!/usr/bin/env python3
"""Compute and display headroom report."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_check import _MockAgent
from src.checker.suite import ConstraintSuite
from src.checker.runner import ConstraintRunner
from src.constraints.base import CheckContext
from src.monitoring.headroom import HeadroomMonitor


def main():
    suite = ConstraintSuite()
    runner = ConstraintRunner(suite, parallel=False)
    agent = _MockAgent()
    context = CheckContext(modification_type="headroom_check")

    verdict = runner.run(agent, context)
    monitor = HeadroomMonitor(warning_threshold=0.05)
    report = monitor.compute_all(verdict)

    print(report.summary())
    print()
    print(monitor.plot_headroom_dashboard(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
