# Mechanistic Interpretability Monitor

## Overview
Monitors model internal representations to detect silent reorganization,
reward hacking, and deceptive alignment during recursive self-improvement.

## Architecture
- **src/probing/**: Probe sets, activation extraction, snapshots, diffs
- **src/attention/**: Head specialization tracking, reward correlation, role tracking
- **src/anomaly/**: Divergence detection, deceptive alignment probes
- **src/integration/**: Hooks for Godel/SOAR/pipeline integration
- **src/monitoring/**: Dashboard, alert rules, time series
- **src/analysis/**: Activation analysis, head evolution, reports

## Commands
- `make install` — install with dev dependencies
- `make test` — run tests
- `make coverage` — run tests with coverage (target >= 90%)
- `make lint` — check syntax
- `make clean` — remove caches

## Design
- Fully self-contained with mock models (no torch/transformers required)
- MockModel returns deterministic numpy activations based on input hash
- All analysis works on numpy arrays for portability
