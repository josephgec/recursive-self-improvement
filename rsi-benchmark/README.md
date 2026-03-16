# End-to-End RSI Benchmarking + Phase 3 Deliverables

Runs the full RSI pipeline across 6 benchmarks for 15 iterations, measuring sustained improvement, collapse resistance, and each paradigm's contribution via 7-condition ablation.

## Quick Start

```bash
pip install -e .
make test    # 183 tests, 96% coverage
python scripts/run_evaluation.py --benchmarks all --iterations 15 --report
python scripts/run_ablation.py --conditions all --repetitions 3 --report
python scripts/run_collapse_comparison.py --report
python scripts/package_deliverables.py --full
```

## 6 Benchmarks

| Benchmark | Tasks | Category | Tests |
|-----------|-------|----------|-------|
| MATH-500 | 32 | Math reasoning | Symbolic equivalence |
| ARC-AGI | 16 | Pattern synthesis | Pixel-exact match |
| OOLONG | 16 | Long-context QA | Semantic match |
| HumanEval | 15 | Code generation | Test execution |
| SWE-bench | 12 | Software engineering | Patch validation |
| Financial | 15 | Domain math | Numeric tolerance |

## Key Questions Answered

1. **Sustained improvement?** — Improvement curves for all 6 benchmarks over 15 iterations
2. **Collapse resistance?** — RSI diverges upward while Phase 0.2 baselines collapse downward
3. **Which paradigm matters?** — 7-condition ablation with contribution waterfall

## 7 Ablation Conditions

| Condition | What's removed |
|-----------|---------------|
| full_pipeline | Nothing (complete system) |
| no_soar | Evolutionary search |
| no_ctm | BDM compactness verification |
| no_godel | Self-modification |
| no_rlm | Recursive context scaling |
| soar_only | Everything except SOAR |
| naive_self_train | All paradigms (collapse baseline) |

## License

MIT
