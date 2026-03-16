"""Weight configuration for GDI composite scoring."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class WeightConfig:
    """Configurable weights for the four drift signals.

    Default weights: semantic=0.30, lexical=0.25, structural=0.20, distributional=0.25
    """
    semantic: float = 0.30
    lexical: float = 0.25
    structural: float = 0.20
    distributional: float = 0.25

    def validate(self) -> bool:
        """Validate that weights sum to 1.0 (within tolerance)."""
        total = self.semantic + self.lexical + self.structural + self.distributional
        return abs(total - 1.0) < 1e-6

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WeightConfig":
        """Create WeightConfig from a config dictionary.

        Args:
            config: Dictionary with optional 'weights' key containing
                    signal weight overrides.

        Returns:
            WeightConfig instance.
        """
        weights = config.get("weights", {})
        wc = cls(
            semantic=weights.get("semantic", 0.30),
            lexical=weights.get("lexical", 0.25),
            structural=weights.get("structural", 0.20),
            distributional=weights.get("distributional", 0.25),
        )
        if not wc.validate():
            raise ValueError(
                f"Weights must sum to 1.0, got "
                f"{wc.semantic + wc.lexical + wc.structural + wc.distributional}"
            )
        return wc

    def as_dict(self) -> Dict[str, float]:
        """Return weights as a dictionary."""
        return {
            "semantic": self.semantic,
            "lexical": self.lexical,
            "structural": self.structural,
            "distributional": self.distributional,
        }
