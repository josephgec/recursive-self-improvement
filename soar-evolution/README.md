# SOAR Evolutionary Search

LLM-powered evolutionary program synthesis for ARC-AGI tasks. The LLM acts as mutation, crossover, and initialization operators in a genetic programming loop, generating Python programs that transform input grids to output grids.

**Target: 52% solve rate on ARC-AGI public evaluation set.**

## Quick Start

```bash
pip install -e .
make test                # 296 tests, 98% coverage
python scripts/run_single_task.py --task-id color_swap --verbose
python scripts/run_benchmark.py --max-tasks 10 --report
```

## Architecture

```
Task → Initialize population → [Evaluate → Select → Mutate/Crossover → Evaluate] × N → Best program
```

| Module | Purpose |
|--------|---------|
| `src/arc/` | Grid data structures, task loading, program evaluation, visualization |
| `src/population/` | Population management, fitness, selection, diversity, elite archive |
| `src/operators/` | LLM initialization (5 variants), mutation (5 types), crossover, error analysis |
| `src/search/` | Evolutionary engine, budget scheduler, early stopping, parallel runner |

## 5 Mutation Types

| Type | When | Temperature |
|------|------|-------------|
| BUG_FIX | Runtime error | 0.2 |
| REFINEMENT | >80% accuracy | 0.3 |
| RESTRUCTURE | <30% accuracy | 0.8 |
| SIMPLIFY | Correct but complex | 0.3 |
| GENERALIZE | Works on some examples | 0.5 |

## License

MIT
