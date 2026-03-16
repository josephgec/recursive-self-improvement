# Prompt Evolution

Evolves system prompts using thinking-model LLMs as genetic operators.

## Quick Start

```bash
pip install -e ".[dev]"
make test-cov
make evolve
```

## Architecture

- `src/genome/` - Prompt genome representation and serialization
- `src/operators/` - Thinking and non-thinking genetic operators
- `src/ga/` - Genetic algorithm engine, population management, selection
- `src/evaluation/` - Financial math benchmarks and answer checking
- `src/comparison/` - Ablation studies and statistical tests
- `src/deliverables/` - Report generation
- `src/analysis/` - Evolution dynamics and prompt analysis

## Testing

```bash
make test          # Run all tests
make test-cov      # Run with coverage report
```

Target: >= 90% code coverage.
