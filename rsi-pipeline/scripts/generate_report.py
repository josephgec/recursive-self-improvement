#!/usr/bin/env python3
"""Generate a pipeline analysis report."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.report import generate_report


def main():
    # Example report with mock data
    pipeline_result = {
        "total_iterations": 10,
        "successful_improvements": 3,
        "rollbacks": 1,
        "emergency_stops": 0,
        "final_accuracy": 0.82,
        "initial_accuracy": 0.70,
        "total_accuracy_gain": 0.12,
        "improvement_rate": 0.3,
        "reason_stopped": "max_iterations",
        "iteration_results": [],
    }

    safety_report = {
        "current_status": {
            "gdi_score": 0.15,
            "car_score": 1.0,
            "constraints_satisfied": True,
            "consecutive_rollbacks": 0,
            "emergency_stop": False,
            "violations": [],
        },
        "gdi_trajectory": {"trend": "stable", "current": 0.15, "max": 0.2, "avg": 0.12},
        "car_trajectory": {"trend": "stable", "current": 1.0, "min": 0.8, "avg": 0.95},
        "violation_summary": {"total": 0, "violations": []},
        "risk_assessment": "low",
    }

    report = generate_report(
        pipeline_result=pipeline_result,
        safety_report=safety_report,
        convergence={"converged": False, "ceiling": 0.90, "marginal_returns": 0.7},
    )
    print(report)


if __name__ == "__main__":
    main()
