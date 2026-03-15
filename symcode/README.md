# SymCode

LLM mathematical reasoning via executable SymPy code generation with verification and self-correction. Instead of prose chain-of-thought, the model outputs Python/SymPy code whose output is deterministically verifiable.

## What It Does

1. **Code-first inference** — Routes math problems to a SymPy code generation pipeline with category-specific few-shot examples.
2. **Self-correction** — Up to k retry attempts with structured error feedback (syntax errors, wrong answers, SymPy-specific issues).
3. **Robust answer checking** — Multi-strategy comparison cascade: exact string, numeric, fraction, symbolic, LaTeX, set matching.
4. **Benchmarking** — Evaluates on MATH-500 and OlympiadBench with head-to-head prose CoT comparison.

**Target: >= 13.6 percentage point improvement over prose baseline on MATH-500.**

## Quick Start

```bash
# Install
pip install -e .

# Run tests (316 tests, ~14 sec)
make test

# Test a single problem
python scripts/evaluate_single.py --problem "Solve x^2 - 4 = 0" --mock

# Debug run (10 problems, mock LLM)
python scripts/run_benchmark.py --benchmark math500 --pipeline both --max-problems 10 --mock

# Full MATH-500 evaluation (requires API key)
export OPENAI_API_KEY=sk-...
python scripts/run_benchmark.py --benchmark math500 --pipeline both --report
```

## Architecture

```
Problem -> Router -> SymCode Generator -> Parser -> Executor -> Answer Check
                         ^                                          |
                         +---- Feedback <---- Error ---------------+ (up to k retries)
```

| Module | Purpose |
|--------|---------|
| `src/pipeline/` | Task routing, prompt assembly, LLM code generation, prose baseline |
| `src/verification/` | Code execution, error classification, feedback, retry loop |
| `src/benchmarks/` | MATH-500 + OlympiadBench loading, runner, metrics, comparison |
| `src/analysis/` | Error taxonomy, difficulty scaling, retry analysis, reports |

## Pipeline Modes

| Mode | Description |
|------|-------------|
| **SymCode** | LLM generates SymPy code, executed in sandbox, answer extracted from `answer` variable |
| **Prose** | Standard chain-of-thought, answer extracted from `\boxed{}` |
| **Hybrid** | Router decides per-problem: SymCode for algebra/calculus/number theory, prose for geometry |

## Self-Correction

When code execution fails, the system classifies the error and generates targeted feedback:

| Error Type | Example Feedback |
|-----------|-----------------|
| `IMPORT_MISSING` | "Add `from sympy import solve` at the top" |
| `NAME_UNDEFINED` | "Variable 'x' used but not defined. Add `x = symbols('x')`" |
| `LOGIC_ERROR` | "Your code produces 42 but expected 28. Check line 5." |
| `TIMEOUT` | "Computation too expensive. Try a simpler approach." |

## Evaluation Metrics

- **Accuracy** — Fraction solved correctly
- **pass@k** — Probability at least one of k samples is correct (unbiased estimator)
- **Retry effectiveness** — Recovery rate: problems initially wrong that self-corrected
- **McNemar's test** — Statistical significance of SymCode vs. prose difference
- **Error taxonomy** — Categorized failure modes across code gen, runtime, logic, self-correction

## Configuration

```yaml
model:
  provider: "openai"       # "openai" | "anthropic" | "mock"
  name: "gpt-4o"
  temperature: 0.0

verification:
  max_retries: 3
  execution_timeout: 30
  numerical_tolerance: 1.0e-6

pipeline:
  router_enabled: true     # Route geometry to prose
```

## Testing

```bash
# All tests (316 tests, 95% coverage)
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## License

MIT
