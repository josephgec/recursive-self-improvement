# SymCode

LLM mathematical reasoning via executable SymPy code generation
with verification and self-correction.

## Quick start
- `make install` -- install deps
- `make test` -- run tests
- `python scripts/evaluate_single.py --problem "Solve x^2 - 4 = 0"` -- test one problem
- `python scripts/run_benchmark.py --benchmark math500 --pipeline both --report` -- full eval

## Architecture
- `src/pipeline/` -- Task routing, prompt assembly, LLM code generation
- `src/verification/` -- Code execution, error classification, feedback, retry loop
- `src/benchmarks/` -- MATH-500 + OlympiadBench loading, benchmark runner, metrics
- `src/analysis/` -- Error taxonomy, difficulty scaling, retry analysis, reports

## Key flow
Problem -> Router -> SymCode Generator -> Parser -> Executor -> Answer Check
                                          ^                       |
                                          +-- Feedback <- Error --+ (up to k retries)

## Target: >=13.6pp improvement over prose baseline on MATH-500
