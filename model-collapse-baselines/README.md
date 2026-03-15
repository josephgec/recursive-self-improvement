# Model Collapse Baselines

Empirically demonstrates model collapse by training recursive model lineages M_0 -> M_1 -> ... -> M_n on controlled mixtures of human and synthetic data. Implements the discrete-time dynamical system P'_t = alpha_t * P + (1 - alpha_t) * Q_t from the RSI paper.

## What It Does

1. **Trains recursive lineages** — Each generation M_t generates synthetic data, which is mixed with real human data at ratio alpha_t, then used to train M_{t+1}.
2. **Measures collapse metrics** — Tracks entropy decay, KL divergence from the original distribution, vocabulary coverage, n-gram diversity, and self-BLEU across generations.
3. **Detects the fixed point Q\*** — Identifies when the model reaches the degenerate fixed point where metrics stabilize but quality has degraded.
4. **Compares across scales** — Runs at 1B and 7B parameter scales to characterize how model size interacts with collapse dynamics.

## Quick Start

```bash
# Install
pip install -e .

# Run tests
make test

# Debug run (~2 min, tiny-gpt2, 3 generations)
python scripts/run_full_experiment.py --scale 1b --schedules zero_alpha --config configs/debug.yaml

# 1B experiment (all schedules, 15 generations)
python scripts/run_full_experiment.py --scale 1b --schedules all

# Full experiment (1B + 7B, all schedules)
python scripts/run_full_experiment.py --scale both --schedules all
```

## Architecture

```
src/
├── data/           # Real data loading, synthetic generation, mixing at alpha_t
├── training/       # Single-gen trainer, lineage orchestrator, alpha schedules
├── measurement/    # Entropy, KL divergence, variance, diversity, tail analysis
└── analysis/       # Collapse curves, scale comparison, phase diagrams, report
```

## Alpha Schedules

The experiment varies the proportion of real data (alpha_t) across four schedules:

| Schedule | Formula | Purpose |
|----------|---------|---------|
| **Constant** | alpha_t = 0.5 | Controlled baseline |
| **Linear decay** | alpha_t: 1.0 -> 0.0 | Simulates gradual data exhaustion |
| **Exponential decay** | alpha_t = gamma^t | Simulates rapid data pollution |
| **Zero** | alpha_t = 0 after M_0 | Worst case: pure self-consumption |

## Collapse Metrics

Each generation is measured on:

- **Train loss** — From the fine-tuning step
- **Perplexity** — Sequence-level entropy of generated text
- **KL divergence** — Divergence from the original human data distribution
- **Distinct-n** — Unique n-grams / total n-grams (n=1,2,3,4)
- **Self-BLEU** — Average BLEU of each text against all others (higher = more repetitive)
- **Vocabulary coverage** — Number of unique tokens used
- **Embedding variance** — Spread of outputs in semantic space
- **Tail mass** — Probability mass on rare tokens (drops during collapse)

## Model Scales

| Config | Model | Parameters | VRAM | Time per generation |
|--------|-------|-----------|------|---------------------|
| `debug.yaml` | sshleifer/tiny-gpt2 | ~2M | <1 GB | ~30 sec |
| `1b_baseline.yaml` | TinyLlama-1.1B | 1.1B | ~8 GB | ~30 min |
| `7b_baseline.yaml` | Mistral-7B (LoRA) | 7B | ~24 GB | ~2 hours |

## Configuration

Configs are in `configs/` as YAML files, merged in order: `default.yaml` < scale config < schedule config < user `--config`.

```yaml
experiment:
  num_generations: 15       # How many M_t to train
  seed: 42

training:
  epochs: 1
  batch_size: 8
  learning_rate: 2.0e-5
  use_lora: false           # True for 7B
  from_pretrained_each_generation: true  # Fresh fine-tune each gen

synthetic_generation:
  num_samples: 50000        # Documents to generate per generation
  temperature: 1.0
  top_p: 0.95
```

## Output

Each lineage produces:

- `checkpoints/generation_XX/` — Model checkpoints per generation
- `metrics/metrics.json` — All collapse metrics per generation
- `analysis/` — Phase diagrams and collapse boundary plots

## Key Design Decisions

- **Fresh fine-tune per generation** (default) — Each M_t starts from the same pretrained base, isolating data quality effects from weight drift. Continual fine-tuning is available via config flag.
- **LoRA for 7B** — Reduces VRAM from 4x A100 to 1x A100. May slightly slow collapse dynamics, which is itself an interesting finding.
- **Unigram KL as primary divergence** — Cheap to compute, directly corresponds to the paper's formalization. Supplemented by embedding-space metrics for semantic-level collapse detection.
- **15 generations** — Prior literature shows collapse within 5-10 generations under aggressive schedules. 15 provides margin to observe both the trajectory and fixed-point stabilization.

## Testing

```bash
# Unit tests (253 tests, ~18 sec)
make test

# With coverage (95%)
pytest tests/ --cov=src --cov-report=term-missing
```

## License

MIT
