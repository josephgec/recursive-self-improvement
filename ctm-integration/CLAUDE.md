# CTM Integration

## Overview
Implements the Block Decomposition Method (BDM) for algorithmic complexity estimation,
symbolic rule synthesis via mock LLM, and a verified rule library with Pareto-optimal selection.

## Project Structure
- `src/bdm/` — CTM table construction, block decomposition, BDM scoring, compression baselines
- `src/synthesis/` — Candidate generation, empirical verification, complexity ranking, Pareto selection
- `src/library/` — Verified rule storage, indexing, composition, evolution
- `src/integration/` — Augmented prompting, comparison, feedback loops
- `src/analysis/` — Complexity landscape, library growth, collapse/escape analysis, reports
- `src/utils/` — Encoding, program length measurement, Turing machine enumeration
- `configs/` — YAML configuration files
- `scripts/` — CLI entry points
- `tests/` — Comprehensive test suite

## Commands
- `make install` — Install in editable mode with dev dependencies
- `make test` — Run tests with coverage
- `make lint` — Lint with ruff
- `make build-ctm-table` — Build the CTM lookup table
- `make run-synthesis` — Run the synthesis loop
- `make run-comparison` — Compare augmented vs standard prompts
- `make report` — Generate analysis report

## Key Concepts
- **BDM**: sum(K_CTM(b_i) + log2(n_i)) where K_CTM is Coding Theorem Method complexity
- **CTM Table**: Pre-computed algorithmic probability for small binary strings via TM enumeration
- **Pareto Selection**: Rules on the accuracy-vs-complexity Pareto front are selected
- **Mock LLM**: CandidateGenerator uses template-based generation for standalone operation

## Testing
All tests run standalone with no external dependencies. Mock LLM is used throughout.
Target >= 90% coverage.
