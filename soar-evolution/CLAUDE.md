# SOAR-Evolution

LLM-powered evolutionary program synthesis for ARC-AGI tasks.

## Quick Start

```bash
make test          # Run tests
make test-cov      # Run tests with coverage
make run-search    # Run evolutionary search
make run-benchmark # Run benchmark suite
```

## Architecture

- `src/arc/` - ARC grid/task representation, loading, evaluation
- `src/population/` - Evolutionary population management
- `src/operators/` - LLM-powered genetic operators (init, mutate, crossover)
- `src/search/` - Search engine, scheduling, early stopping
- `src/analysis/` - Search dynamics analysis and reporting
- `src/utils/` - Shared utilities

## Testing

All tests are offline with mock LLMs. No GPU required.
Target: >= 90% code coverage.
