#!/usr/bin/env python3
"""Run a full SOAR iteration loop."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.iteration.loop import SOARLoop
from src.synthesis.synthesizer import Synthesizer
from src.synthesis.strategies.direct_solution import DirectSolutionStrategy
from src.synthesis.strategies.error_correction import ErrorCorrectionStrategy
from src.synthesis.strategies.improvement_chain import ImprovementChainStrategy
from src.synthesis.strategies.hindsight_relabel import HindsightRelabelStrategy
from src.synthesis.strategies.crossover_pairs import CrossoverPairsStrategy
from src.synthesis.strategies.pattern_description import PatternDescriptionStrategy


def main():
    max_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    synthesizer = Synthesizer()
    synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
    synthesizer.register_strategy(ErrorCorrectionStrategy(), 0.8)
    synthesizer.register_strategy(ImprovementChainStrategy(), 0.7)
    synthesizer.register_strategy(HindsightRelabelStrategy(), 0.5)
    synthesizer.register_strategy(CrossoverPairsStrategy(), 0.6)
    synthesizer.register_strategy(PatternDescriptionStrategy(), 0.4)

    loop = SOARLoop(synthesizer=synthesizer, max_iterations=max_iters)
    history = loop.run()

    print(f"\nCompleted {loop.iteration} iterations")
    print(json.dumps(loop.summary(), indent=2))


if __name__ == "__main__":
    main()
