# RLM - Recursive Language Model

## Overview
RLM wraps LLM completion calls by loading context into a REPL, letting the LLM
interact with the context via Python code.  `rlm.completion()` is a drop-in
replacement for `llm.completion()`.

## Commands
- `make test` — run all tests
- `make coverage` — run tests with coverage report (target >= 90%)
- `make dev` — install in development mode with test dependencies

## Architecture
- `src/core/` — context loading, code execution, session loop, completion API
- `src/recursion/` — sub-query spawning, partitioning, aggregation, depth control
- `src/strategies/` — strategy detection, trajectory logging, REPL helpers
- `src/prompts/` — system prompts that teach the LLM to use the REPL
- `src/evaluation/` — benchmarks (OOLONG, LoCoDiff, synthetic), metrics, runner
- `src/analysis/` — trajectory, cost, depth analysis and reporting
- `src/utils/` — token counting, context generation utilities

## Key design decisions
- Fully self-contained with mock LLMs for offline testing
- All tests are deterministic and require no network
- FINAL("result") protocol signals the LLM is done
