#!/usr/bin/env python3
"""Synthesize training data from trajectories."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collection.collector import TrajectoryCollector
from src.synthesis.synthesizer import Synthesizer
from src.synthesis.strategies.direct_solution import DirectSolutionStrategy
from src.synthesis.strategies.error_correction import ErrorCorrectionStrategy
from src.synthesis.strategies.improvement_chain import ImprovementChainStrategy
from src.synthesis.strategies.hindsight_relabel import HindsightRelabelStrategy
from src.synthesis.strategies.crossover_pairs import CrossoverPairsStrategy
from src.synthesis.strategies.pattern_description import PatternDescriptionStrategy
from src.synthesis.quality_filter import QualityFilter
from src.synthesis.deduplicator import Deduplicator
from src.synthesis.formatter import Formatter


def main():
    trajectory_dir = sys.argv[1] if len(sys.argv) > 1 else "data/trajectories"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/training_data/train.jsonl"

    # Collect
    collector = TrajectoryCollector(trajectory_dir=trajectory_dir)
    trajectories = collector.collect_from_directory()
    print(f"Loaded {len(trajectories)} trajectories")

    # Synthesize
    synthesizer = Synthesizer()
    synthesizer.register_strategy(DirectSolutionStrategy(), 1.0)
    synthesizer.register_strategy(ErrorCorrectionStrategy(), 0.8)
    synthesizer.register_strategy(ImprovementChainStrategy(), 0.7)
    synthesizer.register_strategy(HindsightRelabelStrategy(), 0.5)
    synthesizer.register_strategy(CrossoverPairsStrategy(), 0.6)
    synthesizer.register_strategy(PatternDescriptionStrategy(), 0.4)

    pairs = synthesizer.synthesize(trajectories)
    print(f"Synthesized {len(pairs)} training pairs")

    # Filter
    qf = QualityFilter()
    pairs = qf.filter(pairs)
    print(f"After filtering: {len(pairs)} pairs")

    # Deduplicate
    dedup = Deduplicator()
    pairs = dedup.deduplicate(pairs)
    print(f"After dedup: {len(pairs)} pairs")

    # Format and write
    formatter = Formatter(format_type="openai_jsonl")
    n_written = formatter.write_jsonl(pairs, output_file)
    print(f"Wrote {n_written} entries to {output_file}")


if __name__ == "__main__":
    main()
