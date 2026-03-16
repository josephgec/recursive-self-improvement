# Publication Ablation Studies

31 conditions across 4 suites isolating every component's contribution to the RSI pipeline. Generates paper-ready LaTeX tables, publication-quality figures, and draft results section prose.

## Quick Start

```bash
pip install -e .
make test    # 193 tests, 96% coverage
python scripts/run_all_suites.py --generate-paper-assets --report
python scripts/generate_paper_assets.py --suite all --output-dir paper/figures/
```

## 4 Suites

| Suite | Conditions | Paper | Primary Benchmarks |
|-------|-----------|-------|-------------------|
| Neurosymbolic | 7 | SymCode + BDM | MATH-500, HumanEval |
| Gödel Agent | 8 | Self-modification safety | MATH-500, HumanEval, ARC-AGI |
| SOAR | 8 | Evolutionary operators | ARC-AGI, MATH-500 |
| RLM | 8 | Recursive REPL | OOLONG, MATH-500 |

## Output

- **LaTeX tables** with `\toprule`/`\midrule`/`\bottomrule`, significance stars (\*, \*\*, \*\*\*), bold best
- **PDF figures** — bar charts, forest plots, improvement curves, contribution waterfalls
- **Draft narrative** — results section prose with inline CI, p-values, and effect sizes
- **Statistical rigor** — paired t-tests, Bonferroni correction, Cohen's d, bootstrap 95% CI, power analysis

## License

MIT
