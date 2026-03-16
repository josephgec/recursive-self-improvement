#!/usr/bin/env python3
"""Generate a comprehensive SOAR report."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.report import ReportGenerator
from src.iteration.loop import SOARLoop
from src.synthesis.synthesizer import Synthesizer
from src.synthesis.strategies.direct_solution import DirectSolutionStrategy
from src.synthesis.strategies.error_correction import ErrorCorrectionStrategy


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/reports"

    # Run a quick loop to generate data for the report
    synthesizer = Synthesizer()
    synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
    synthesizer.register_strategy(ErrorCorrectionStrategy(), 0.8)

    loop = SOARLoop(synthesizer=synthesizer, max_iterations=3)
    history = loop.run()

    # Generate report
    generator = ReportGenerator(output_dir=output_dir)
    report = generator.generate(
        pairs=loop.synthesizer.pairs,
        iteration_history=history,
    )

    # Save
    filepath = generator.save(report)
    print(f"Report saved to {filepath}")
    print(generator.format_text(report))


if __name__ == "__main__":
    main()
