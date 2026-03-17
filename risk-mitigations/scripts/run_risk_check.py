#!/usr/bin/env python3
"""Run a complete risk check across all 6 domains."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.orchestration.risk_registry import RiskRegistry
from src.orchestration.risk_dashboard import UnifiedRiskDashboard


def main():
    registry = RiskRegistry()
    dashboard = UnifiedRiskDashboard(registry)
    result = dashboard.compute()
    report = dashboard.generate_stakeholder_report(result)
    print(report)
    return 0 if not result.needs_immediate_action else 1


if __name__ == "__main__":
    sys.exit(main())
