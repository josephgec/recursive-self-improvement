# RSI Design Experiments

Controlled experiments resolving 3 critical RSI pipeline design decisions: modification frequency, hindsight target, and RLM recursion depth. Each produces a statistically validated recommendation with confidence intervals.

## Quick Start

```bash
pip install -e .
make test    # 165 tests, 98% coverage
python scripts/run_all_experiments.py --report
python scripts/generate_optimal_config.py
```

## 3 Experiments

| Experiment | Conditions | Key Question |
|-----------|-----------|-------------|
| Modification frequency | 7 (every_task → never) | How often should the agent self-modify? |
| Hindsight target | 6 (weights/library/both/none) | Where should learning signal go? |
| RLM depth | 7 (depth 0-6) | Optimal recursion depth before diminishing returns? |

## Analysis

- One-way ANOVA per experiment with Tukey HSD pairwise comparisons
- Sensitivity ranking across all 3 experiments
- Diminishing returns detection (Kneedle algorithm for depth)
- Optimal config YAML export for Phase 3.1 pipeline

## License

MIT
