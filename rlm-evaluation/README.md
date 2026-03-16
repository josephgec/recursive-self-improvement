# RLM Evaluation + Phase 2b Deliverables

Benchmarks the Recursive Language Model on OOLONG, LoCoDiff, and synthetic long-context tasks. Analyzes emergent strategies and validates RLM(small) vs standard(frontier) cost-efficiency claims.

## Quick Start

```bash
pip install -e .
make test    # 182 tests, 98% coverage
python scripts/run_all_benchmarks.py --benchmarks all --report
python scripts/run_scaling_experiment.py --max-tokens 10000000
python scripts/run_comparison.py --package-deliverables
```

## Key Experiments

1. **OOLONG + LoCoDiff accuracy** — per-category breakdown
2. **Context scaling** — accuracy flat for RLM, declining for standard
3. **RLM(small) beats standard(frontier)** — 2x on correct answers at lower cost
4. **Emergent strategy analysis** — grep-first and map-reduce emerge naturally

## 6 Emergent Strategies

| Strategy | When it emerges | Signature |
|----------|----------------|-----------|
| DIRECT | Small context (<10K) | 1-2 code blocks, immediate FINAL |
| PEEK_THEN_GREP | Retrieval tasks | peek → grep → slice → FINAL |
| ITERATIVE_SEARCH | Exploratory queries | 3+ narrowing grep calls |
| MAP_REDUCE | Aggregation/counting | chunk → sub-query per chunk → aggregate |
| HIERARCHICAL | Very large context | Nested sub-queries (depth > 1) |
| HYBRID | Complex reasoning | Multiple strategies combined |

## License

MIT
