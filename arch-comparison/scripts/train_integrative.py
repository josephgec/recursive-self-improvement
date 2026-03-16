#!/usr/bin/env python3
"""Train the integrative model (mock)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integrative.training import IntegrativeTrainer


def main() -> None:
    print("Training integrative model (mock)...")
    trainer = IntegrativeTrainer(
        hidden_dim=64,
        num_heads=4,
        learning_rate=0.001,
        epochs=5,
    )

    model = trainer.prepare_model()
    print(f"Model prepared: {sum(p.numel() for p in model.parameters())} parameters")

    result = trainer.train()
    print(f"Training completed: {result.epochs_completed} epochs")
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Loss history: {[f'{l:.4f}' for l in result.loss_history]}")

    accuracy = trainer.evaluate()
    print(f"Evaluation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
