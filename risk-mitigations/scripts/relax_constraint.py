#!/usr/bin/env python3
"""Propose and apply a graduated constraint relaxation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.constraints.graduated_relaxation import GraduatedRelaxation


def main():
    relaxation = GraduatedRelaxation()
    relaxation.set_constraint("quality_threshold", 0.90)
    relaxation.set_constraint("no_harmful_output", 1.0)

    # Try relaxing quality threshold
    proposal = relaxation.propose_relaxation("quality_threshold", "Performance bottleneck")
    print(f"Proposal for quality_threshold:")
    print(f"  Approved: {proposal.approved}")
    print(f"  Current: {proposal.current_value:.2f}")
    print(f"  Proposed: {proposal.proposed_value:.2f}")
    print(f"  Step: {proposal.step_number}/{proposal.max_steps}")

    if proposal.approved:
        relaxation.apply_relaxation(proposal)
        print(f"  Applied! New value: {relaxation.get_constraint_value('quality_threshold'):.2f}")

    # Try relaxing safety constraint
    safety_proposal = relaxation.propose_relaxation("no_harmful_output")
    print(f"\nProposal for no_harmful_output:")
    print(f"  Approved: {safety_proposal.approved}")
    print(f"  Reason: {safety_proposal.reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
