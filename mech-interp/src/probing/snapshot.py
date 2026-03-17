"""Save and load activation snapshots as JSON."""

import json
import os
from typing import Optional

from src.probing.extractor import ActivationSnapshot


def save_snapshot(snapshot: ActivationSnapshot, path: str) -> None:
    """Save an activation snapshot to a JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    data = snapshot.to_dict()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_snapshot(path: str) -> ActivationSnapshot:
    """Load an activation snapshot from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return ActivationSnapshot.from_dict(data)
