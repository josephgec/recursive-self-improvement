#!/usr/bin/env python3
"""Collect and process search trajectories."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collection.collector import TrajectoryCollector
from src.collection.database import TrajectoryDatabase
from src.collection.indexer import TrajectoryIndexer
from src.collection.statistics import CorpusStatistics


def main():
    trajectory_dir = sys.argv[1] if len(sys.argv) > 1 else "data/trajectories"

    collector = TrajectoryCollector(trajectory_dir=trajectory_dir)
    trajectories = collector.collect_from_directory()
    print(f"Collected {len(trajectories)} trajectories")

    # Index
    indexer = TrajectoryIndexer()
    indexer.index_many(trajectories)
    print(f"Indexed {indexer.total_indexed()} trajectories")

    # Extract improvement chains
    chains = collector.extract_improvement_chains()
    print(f"Extracted {len(chains)} improvement chains")

    # Statistics
    stats = CorpusStatistics(trajectories)
    report = stats.full_report()
    print(json.dumps(report, indent=2))

    # Store
    db = TrajectoryDatabase(db_dir=trajectory_dir)
    for traj in trajectories:
        db.store(traj)
    print(f"Stored {db.count()} trajectories in database")


if __name__ == "__main__":
    main()
