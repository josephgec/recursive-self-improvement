# Ablation Studies

Paper-ready ablation studies with 31 conditions across 4 suites.

## Quick Start

```bash
make test        # Run all tests
make coverage    # Run with coverage report (target >= 90%)
make run-all     # Run all ablation suites
make paper-assets # Generate publication figures/tables
```

## Architecture

- `src/suites/` - Suite definitions (neurosymbolic, godel, soar, rlm)
- `src/conditions/` - Per-condition pipeline configuration builders
- `src/execution/` - Runner, parallelism, checkpointing
- `src/analysis/` - Statistical tests, effect sizes, power analysis
- `src/publication/` - LaTeX tables, figures, narrative generation
- `src/utils/` - Reproducibility, cost estimation

## Key Design Decisions

- All tests use MockPipeline with deterministic accuracy per condition
- Statistical tests use paired t-tests with Bonferroni correction
- Publication output uses booktabs LaTeX style
- Figures are colorblind-safe with serif fonts
