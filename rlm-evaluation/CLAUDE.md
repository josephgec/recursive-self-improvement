# RLM Evaluation

Benchmarking framework for the RLM (Recursive Language Model) system on long-context tasks.

## Quick Start

```bash
pip install -e ".[dev]"
make test
make coverage
```

## Structure

- `src/benchmarks/` - Task definitions, benchmark loaders (OOLONG, LoCoDiff, synthetic)
- `src/execution/` - RLM and standard executor, runner, checkpointing
- `src/strategies/` - Strategy classification, emergence analysis, effectiveness
- `src/comparison/` - Cost models, head-to-head, scaling experiments
- `src/deliverables/` - Phase 2b report packaging
- `src/analysis/` - Visualization and report generation
- `configs/` - YAML configurations
- `scripts/` - CLI entry points
- `tests/` - Comprehensive test suite (target >= 90% coverage)

## Key Commands

- `make test` - Run tests
- `make coverage` - Run with coverage report
- `make run-all` - Run all benchmarks
- `make run-scaling` - Run scaling experiment
- `make report` - Generate analysis report
