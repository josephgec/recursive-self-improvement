# CTM Integration

Algorithmic probability via the Block Decomposition Method (BDM) for building a verified symbolic rule library that escapes model collapse. Instead of training on token predictions (which collapses), accumulates a library of verified symbolic rules scored by algorithmic complexity.

## Core Idea

Standard LLM training minimizes cross-entropy loss, bounded by Shannon entropy — it can only learn statistically present patterns. BDM-guided synthesis escapes this bound by finding *mechanistic* explanations: short programs that generate observed patterns. A model with verified rules doesn't collapse from recursive self-training because rules are verified independently of the training distribution.

## What It Does

1. **BDM complexity scoring** — Approximates Kolmogorov complexity via the Coding Theorem Method, decomposing data into blocks and scoring each block's algorithmic probability.
2. **Symbolic rule synthesis** — LLM generates candidate programmatic rules, BDM scores them for simplicity, empirical tests verify accuracy, Pareto-optimal rules are retained.
3. **Rule library** — Growing collection of verified rules that the LLM invokes during reasoning, creating an alternative self-improvement pathway.

## Quick Start

```bash
pip install -e .

# Run tests (219 tests, ~2 sec)
make test

# Build CTM lookup table (one-time)
python scripts/build_ctm_table.py --block-size 8

# Run synthesis loop
python scripts/run_synthesis.py --domain math --iterations 20 --report

# Compare augmented vs standard LLM
python scripts/run_comparison.py --library-path data/rules/
```

## Architecture

```
src/
├── bdm/           # CTM table, block decomposition, BDM scoring, calibration
├── synthesis/     # Candidate generation, verification, Pareto selection
├── library/       # Rule storage, indexing, composition, evolution
├── integration/   # Augmented prompting, rule invocation, comparison
└── analysis/      # Complexity landscape, library growth, collapse escape
```

## BDM Scoring

BDM(x) = Sum_i [K_CTM(b_i) + log2(n_i)]

Where b_i are unique blocks and n_i their multiplicities. Repeated blocks contribute log2(count) instead of full complexity — capturing that repetition implies a short generator.

```python
from src.bdm.scorer import BDMScorer
from src.bdm.ctm_table import CTMTable

table = CTMTable.build(max_states=2, block_size=8)
scorer = BDMScorer(table)

# Repetitive data: low BDM
score_rep = scorer.score("01010101" * 10)

# Random data: high BDM
score_rand = scorer.score("10110100" * 10)

assert score_rep.normalized_score < score_rand.normalized_score
```

## Synthesis Loop

Each iteration: generate candidates -> verify empirically -> score with BDM -> select Pareto front -> refine top rules -> add to library.

```python
from src.synthesis.synthesis_loop import SymbolicSynthesisLoop

loop = SymbolicSynthesisLoop(generator, verifier, selector, library, config)
result = loop.run(training_data, max_iterations=50)
# result.final_library_size, result.final_pareto_front
```

## Rule Library

Rules are Pareto-optimal in (accuracy, simplicity) space:

| Rule | Accuracy | BDM | Domain |
|------|----------|-----|--------|
| `f(n) = n**2` | 100% | 12 bits | math |
| `fib(n) = fib(n-1) + fib(n-2)` | 100% | 18 bits | sequences |
| `f(x) = 2*x + 1` | 100% | 10 bits | math |

## Why This Escapes Model Collapse

- Rules are verified *independently* of training distribution
- A rule either passes empirical tests or it doesn't — no distributional drift
- Library grows monotonically (modulo pruning of dominated rules)
- New rules ADD information; they never corrupt existing verified knowledge
- BDM ensures rules are mechanistic, not statistical artifacts

## Testing

```bash
# 219 tests, 94% coverage
pytest tests/ -v --tb=short
pytest tests/ --cov=src --cov-report=term-missing
```

## License

MIT
