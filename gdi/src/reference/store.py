"""Reference output storage with JSON persistence."""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional


class ReferenceStore:
    """Persistent storage for reference outputs.

    Saves and loads reference outputs as JSON files, with support
    for archiving old references on update.
    """

    def __init__(self, store_path: str):
        """Initialize store.

        Args:
            store_path: Path to the JSON file for storage.
        """
        self.store_path = store_path
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure the parent directory exists."""
        parent = os.path.dirname(self.store_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def save(self, data: Dict[str, Any]) -> None:
        """Save reference data to JSON file.

        Args:
            data: Dictionary containing reference outputs and metadata.
        """
        data["saved_at"] = datetime.now().isoformat()
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> Dict[str, Any]:
        """Load reference data from JSON file.

        Returns:
            Dictionary containing reference outputs and metadata.

        Raises:
            FileNotFoundError: If no reference data exists.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"No reference data at {self.store_path}"
            )
        with open(self.store_path, "r") as f:
            return json.load(f)

    def exists(self) -> bool:
        """Check if reference data exists."""
        return os.path.exists(self.store_path)

    def update(self, new_data: Dict[str, Any]) -> None:
        """Update reference data, archiving the old version.

        Args:
            new_data: New reference data to save.
        """
        if self.exists():
            # Archive old reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.store_path.replace(
                ".json", f"_archive_{timestamp}.json"
            )
            shutil.copy2(self.store_path, archive_path)

        self.save(new_data)
