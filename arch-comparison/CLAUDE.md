# CLAUDE.md — arch-comparison

## Project purpose
Compare three neural architectures for RSI suitability:
- **Hybrid**: LLM + external solvers (SymPy, Z3) via tool-calling
- **Integrative**: LNN-style constrained decoding with logical attention
- **Prose baseline**: Plain LLM with no augmentation

Evaluation axes: generalization, interpretability, robustness.

## Commands
- `make test` — run all tests
- `make test-cov` — run tests with coverage report
- `python scripts/run_comparison.py` — full 3-way comparison
- `python scripts/run_single_axis.py --axis generalization` — single axis

## Architecture
- `src/hybrid/` — tool-calling pipeline with chain logging
- `src/integrative/` — constrained decoder, LNN attention, logical loss
- `src/evaluation/` — generalization, interpretability, robustness evaluators
- `src/analysis/` — head-to-head, failure modes, cost, RSI suitability
- `src/deliverables/` — report packaging
- `src/utils/` — task domains, perturbations

## Key design decisions
- All LLM calls use mock deterministic responses (no API keys needed)
- No GPU required — torch CPU only
- Self-contained: no external services
- Statistical tests use scipy (McNemar, chi-squared)
