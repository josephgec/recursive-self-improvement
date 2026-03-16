# Architecture Comparison + Phase 1a Deliverables

3-way comparison of hybrid (LLM + external solver) vs integrative (LNN-style constrained decoding) vs prose baseline, evaluated across generalization, interpretability, and robustness. Consolidates all Phase 1a deliverables.

## What It Does

1. **Hybrid pipeline** — LLM reasons in natural language, delegates formal operations to external SymPy/Z3 solvers via tool calling.
2. **Integrative pipeline** — Symbolic constraints encoded into the neural architecture: constrained decoding, logical loss terms, Logic Neural Network attention layer.
3. **3-axis evaluation** — Generalization (cross-domain transfer), interpretability (step verifiability, faithfulness, readability), robustness (perturbation resistance).
4. **RSI suitability assessment** — Scores each architecture on modularity, verifiability, composability, contamination resistance, and transparency.

## Quick Start

```bash
pip install -e .
make test            # 217 tests, 97% coverage
python scripts/run_comparison.py --axes all --report
python scripts/package_deliverables.py  # Assemble Phase 1a report
```

## Three Evaluation Axes

| Axis | What it measures | Expected winner |
|------|-----------------|----------------|
| **Generalization** | Train on algebra/calculus, test on number theory/logic | Hybrid |
| **Interpretability** | Step verifiability, faithfulness, readability | Hybrid |
| **Robustness** | Rephrase, noise, domain shift, adversarial | Hybrid |

## Testing

```bash
pytest tests/ -v --tb=short  # 217 tests, 97% coverage
```

## License

MIT
