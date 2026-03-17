# Reward Hacking Defense System

## Overview
Three defense mechanisms against reward hacking in RL-based LLM training,
plus safety deliverable packaging. Fully self-contained with mocks -- no
torch or transformers required.

## Structure
- `src/eppo/` -- Entropy-Penalized PPO training with adaptive entropy bonuses
- `src/bounding/` -- Reward clipping, delta bounding, normalization, monitoring
- `src/energy/` -- Activation energy tracking and homogenization detection
- `src/detection/` -- Reward-accuracy divergence, shortcut detection, composite checks
- `src/integration/` -- SOAR adapter, pipeline adapter, training wrapper
- `src/deliverables/` -- Phase gate safety packages, reports, cross-signal analysis
- `src/analysis/` -- Analysis utilities for each subsystem
- `configs/` -- YAML configuration files
- `scripts/` -- Runnable scripts for each subsystem
- `tests/` -- Pytest test suite (target >= 90% coverage)

## Commands
- `make test` -- run test suite
- `make coverage` -- run with coverage report
- `make run-eppo` -- run EPPO training demo
- `make monitor-energy` -- run energy monitoring demo
- `make check-hacking` -- run reward hacking checks
- `make package-safety` -- generate safety package
- `make report` -- generate full report

## Design Decisions
- All models are mocked with numpy -- no ML framework dependencies
- Entropy bonus supports both coefficient-decay and adaptive-target modes
- Reward bounding pipeline: normalize -> clip -> delta-bound
- Energy tracking uses L2 norms of activation arrays
- Safety packages require all four tracks (GDI, Constraint, Interp, Reward) green
