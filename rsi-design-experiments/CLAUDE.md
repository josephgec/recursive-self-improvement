# RSI Design Experiments

Controlled experiments on 3 RSI pipeline design decisions:
1. Modification frequency
2. Hindsight target
3. RLM recursion depth

## Commands
- `make test` - Run all tests
- `make coverage` - Run tests with coverage report
- `make run-all` - Run all experiments
- `make generate-config` - Generate optimal configuration

## Structure
- `src/experiments/` - Experiment definitions (base, frequency, hindsight, depth)
- `src/conditions/` - Condition dataclasses for each experiment
- `src/measurement/` - Trackers (accuracy, stability, cost, improvement rate, composite)
- `src/analysis/` - Statistical analysis (ANOVA, interactions, diminishing returns, etc.)
- `src/harness/` - Execution harness (runner, controlled pipeline, checkpoints, parallel)
- `src/reporting/` - Report generation and config export
- `configs/` - YAML configuration files
- `scripts/` - Entry-point scripts
- `tests/` - Test suite (target >= 90% coverage)

## Notes
- Fully self-contained with mocks; no external RSI pipeline needed.
- Deterministic seeds for reproducibility.
- MockPipeline simulates accuracy changes based on design decisions.
