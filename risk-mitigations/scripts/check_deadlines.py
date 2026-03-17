#!/usr/bin/env python3
"""Check publication deadline statuses."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.publication.deadline_tracker import DeadlineTracker


def main():
    tracker = DeadlineTracker()
    statuses = tracker.check()

    print("Publication Deadlines:")
    for s in statuses:
        marker = "[!]" if s.is_urgent else "[ ]"
        print(f"  {marker} {s.name}: {s.severity}")
        print(f"      Abstract: {s.abstract_date} ({s.days_to_abstract} days)")
        print(f"      Submission: {s.submission_date} ({s.days_to_submission} days)")

    next_dl = tracker.next_deadline()
    if next_dl:
        print(f"\nNext deadline: {next_dl.name} ({next_dl.days_to_submission} days)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
