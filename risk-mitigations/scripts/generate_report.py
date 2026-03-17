#!/usr/bin/env python3
"""Generate a comprehensive risk management report."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.orchestration.risk_registry import RiskRegistry
from src.orchestration.risk_dashboard import UnifiedRiskDashboard
from src.orchestration.incident_manager import IncidentManager
from src.analysis.report import ReportGenerator


def main():
    # Set up
    registry = RiskRegistry()
    dashboard = UnifiedRiskDashboard(registry)
    incident_mgr = IncidentManager()

    # Compute dashboard
    db = dashboard.compute()

    # Generate report
    report_gen = ReportGenerator(title="Comprehensive Risk Report")
    report_gen.add_section(
        "Summary",
        "This report covers all 6 risk domains for the current iteration.",
    )

    report = report_gen.generate(
        dashboard=db,
        incidents=incident_mgr.get_history(),
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
