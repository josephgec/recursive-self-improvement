#!/usr/bin/env python3
"""Run EPPO training demo."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eppo.config import EPPOConfig
from src.eppo.trainer import EPPOTrainer


def main():
    config = EPPOConfig(
        entropy_mode="target",
        entropy_target=2.0,
        entropy_coeff=0.02,
        epochs=5,
    )
    trainer = EPPOTrainer(config)

    print("Starting EPPO Training")
    print("=" * 50)

    for epoch in range(config.epochs):
        result = trainer.train_epoch(num_steps=10)
        print(
            f"Epoch {epoch}: "
            f"loss={result.mean_combined_loss:.4f}, "
            f"entropy={result.mean_entropy:.4f}, "
            f"beta={result.final_beta:.6f}"
        )

    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    print(f"  Total Steps: {summary['total_steps']}")
    print(f"  Final Entropy: {summary['final_entropy']:.4f}")
    print(f"  Final Beta: {summary['final_beta']:.6f}")


if __name__ == "__main__":
    main()
