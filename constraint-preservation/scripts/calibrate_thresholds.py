#!/usr/bin/env python3
"""Calibrate thresholds based on tightness analysis."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.constraint_tightness import ConstraintTightnessAnalyzer


def main():
    # Simulate some audit entries for calibration
    sample_entries = []
    for i in range(20):
        entry = {
            "passed": i % 3 != 0,
            "violations": ["accuracy_floor"] if i % 3 == 0 else [],
            "modification_type": "code_change",
            "results_summary": {
                "accuracy_floor": {
                    "satisfied": i % 3 != 0,
                    "measured_value": 0.82 if i % 3 != 0 else 0.78,
                    "threshold": 0.80,
                    "headroom": 0.02 if i % 3 != 0 else -0.02,
                },
                "safety_eval": {
                    "satisfied": True,
                    "measured_value": 1.0,
                    "threshold": 1.0,
                    "headroom": 0.0,
                },
            },
        }
        sample_entries.append(entry)

    analyzer = ConstraintTightnessAnalyzer(sample_entries)
    analysis = analyzer.analyze()
    suggestions = analyzer.suggest_adjustments()

    print("Tightness Analysis:")
    for name, info in analysis.items():
        print(f"  {name}: {info['assessment']} (violation_rate={info['violation_rate']:.2f})")

    if suggestions:
        print("\nSuggested Adjustments:")
        for s in suggestions:
            print(f"  {s['constraint']}: {s['suggestion']} ({s['issue']}, rate={s['violation_rate']:.2f})")
    else:
        print("\nAll constraints are well calibrated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
