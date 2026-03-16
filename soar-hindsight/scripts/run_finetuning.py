#!/usr/bin/env python3
"""Run fine-tuning on synthesized training data."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finetuning.trainer import Trainer
from src.finetuning.data_loader import DataLoader
from src.synthesis.synthesizer import TrainingPair


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/training_data/train.jsonl"
    backend = sys.argv[2] if len(sys.argv) > 2 else "openai"

    loader = DataLoader()
    raw = loader.load_from_jsonl(input_file)
    pairs = loader.load_from_dicts(raw)
    print(f"Loaded {len(pairs)} training pairs")
    print(json.dumps(loader.stats(), indent=2))

    trainer = Trainer(backend=backend)
    result = trainer.train(pairs)
    print(f"\nTraining complete:")
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
    print(f"Final train loss: {result.get('metrics', {}).get('train_loss_final', 'N/A')}")


if __name__ == "__main__":
    main()
