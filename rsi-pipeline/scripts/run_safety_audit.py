#!/usr/bin/env python3
"""Run a safety audit on the pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.state import PipelineState, AgentCodeSnapshot
from src.safety.gdi_monitor import GDIMonitor
from src.safety.constraint_enforcer import ConstraintEnforcer
from src.safety.car_tracker import CARTracker
from src.safety.emergency_stop import EmergencyStop
from src.analysis.safety_report import SafetyReportGenerator

import json


def main():
    initial_code = "def solve(x): return x + 1"
    state = PipelineState(
        agent_code=AgentCodeSnapshot(code=initial_code),
        original_code=AgentCodeSnapshot(code=initial_code),
    )
    state.performance.accuracy = 0.7

    gdi = GDIMonitor()
    gdi_score = gdi.compute(state.agent_code.code, state.original_code.code)
    state.safety.gdi_score = gdi_score

    enforcer = ConstraintEnforcer()
    verdict = enforcer.check_all(state)

    car = CARTracker()
    car_score = car.compute(0.65, 0.7)
    state.safety.car_score = car_score

    estop = EmergencyStop()
    emergency = estop.check(state)

    report_gen = SafetyReportGenerator()
    report_gen.set_gdi_history([gdi_score])
    report_gen.set_car_history([car_score])
    report = report_gen.generate_safety_report(state)

    print(json.dumps(report, indent=2))
    print(f"\nRisk assessment: {report['risk_assessment']}")


if __name__ == "__main__":
    main()
