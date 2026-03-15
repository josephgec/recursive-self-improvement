# Model Collapse Baselines

Empirically demonstrates model collapse by training recursive model lineages
M_0 -> M_1 -> ... -> M_n on controlled mixtures of human and synthetic data.

## Quick start
- `make install` -- install dependencies
- `make test` -- run tests
- `python scripts/run_full_experiment.py --scale 1b --schedules zero_alpha --config configs/debug.yaml` -- fast test
- `python scripts/run_full_experiment.py --scale both --schedules all` -- full production run

## Architecture
- `src/data/` -- Real data loading, synthetic generation, mixing at ratio alpha_t
- `src/training/` -- Single-gen trainer, lineage orchestrator, alpha schedules
- `src/measurement/` -- Entropy, KL divergence, variance, diversity, tail analysis, fixed point detection
- `src/analysis/` -- Collapse curves, scale comparison, phase diagrams, report

## Key config flags
- `training.from_pretrained_each_generation: true` -- fresh fine-tune (default)
- `training.use_lora: true` -- required for 7B scale
- `experiment.num_generations: 15` -- how many M_t to train

## Experiment matrix
4 alpha schedules x 2 scales (1B, 7B) x 15 generations = 120 training runs.
