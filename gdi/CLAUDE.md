# GDI — Goal Drift Index

A 4-signal drift detection system for monitoring AI agent behavior drift.

## Quick Start

```bash
make install
make test
make coverage
```

## Architecture

- **src/signals/** — Four drift signals: semantic, lexical, structural, distributional
- **src/composite/** — GDI composite score computation with configurable weights
- **src/reference/** — Reference output collection and storage
- **src/calibration/** — Threshold calibration from collapse data
- **src/integration/** — Hooks, decorators, and phase adapters
- **src/alerting/** — Alert management and escalation policies
- **src/monitoring/** — Time series tracking and anomaly detection
- **src/analysis/** — Signal decomposition, drift characterization, reporting

## Key Design Decisions

- No external ML model downloads needed — semantic signal uses word-frequency cosine distance
- All signals are deterministic and fast
- Regex-based structural parsing (no spacy dependency)
- JSON persistence for reference stores and time series
- Human-gated reference updates

## Testing

All tests run offline without model downloads. Target >= 90% coverage.

```bash
make coverage
```

## Config Files

- `configs/default.yaml` — Standard thresholds and weights
- `configs/strict.yaml` — Lower thresholds for safety-critical deployments
- `configs/relaxed.yaml` — Higher thresholds for exploratory phases
