#!/usr/bin/env python3
"""Generate a full analysis report."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_check import _MockAgent
from src.checker.suite import ConstraintSuite
from src.checker.runner import ConstraintRunner
from src.constraints.base import CheckContext
from src.enforcement.audit import ConstraintAuditLog
from src.monitoring.trend import TrendDetector
from src.analysis.report import generate_report


def main():
    suite = ConstraintSuite()
    runner = ConstraintRunner(suite, parallel=False)
    agent = _MockAgent()
    context = CheckContext(modification_type="report_generation")

    verdict = runner.run(agent, context)

    # Build a small audit history
    audit = ConstraintAuditLog()
    audit.log(verdict, context, "allowed" if verdict.passed else "rejected")

    # Build trend data
    detector = TrendDetector(window_size=5)
    for name, result in verdict.results.items():
        detector.record({name: result.headroom})

    report = generate_report(
        verdict=verdict,
        audit_entries=audit.get_history(),
        trend_detector=detector,
    )

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
