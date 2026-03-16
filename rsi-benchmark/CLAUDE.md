# RSI Benchmark

Recursive Self-Improvement benchmark suite. Runs the full RSI pipeline across
6 benchmarks for 15 iterations, measures sustained improvement, compares to
collapse baselines, and runs 7-condition paradigm ablation.

## Commands
- `make test` - run all tests
- `make test-cov` - run tests with coverage
- `make run-eval` - run full evaluation pipeline
- `make run-ablation` - run ablation study
- `make run-report` - generate final report

## Structure
- `src/benchmarks/` - 6 benchmark implementations with built-in tasks
- `src/evaluation/` - iteration evaluation, improvement curves, held-out
- `src/collapse/` - collapse baselines, divergence, entropy, sustainability
- `src/ablation/` - 7-condition paradigm ablation study
- `src/deliverables/` - report packaging
- `src/analysis/` - cross-benchmark, scaling, cost, qualitative analysis

## Testing
All tests are fully self-contained with mock agents. No external dependencies.
Target >= 90% coverage.
