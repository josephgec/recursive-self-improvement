# Goal Drift Index (GDI)

Multi-signal drift detection for the RSI pipeline. Detects when the system silently drifts from intended behavior across 4 independent dimensions, before accuracy metrics show degradation.

## Quick Start

```bash
pip install -e .
make test    # 195 tests, 98% coverage
python scripts/collect_reference.py --agent-config agent.yaml
python scripts/compute_gdi.py --reference data/reference/
```

## 4 Drift Signals

| Signal | Detects | Sub-metrics |
|--------|---------|-------------|
| Semantic | Meaning changes | Centroid distance, MMD, pairwise |
| Lexical | Vocabulary changes | JS divergence, vocab shift, n-gram novelty |
| Structural | Format changes | Sentence length, depth, POS categories |
| Distributional | Token dist changes | KL (forward/reverse), TV, JS |

## Thresholds (calibrated from Phase 0.2 collapse data)

- Green < 0.15: healthy
- Yellow 0.15-0.25: warning
- Orange 0.25-0.35: concern
- Red > 0.50: critical (pipeline paused)

## License

MIT
