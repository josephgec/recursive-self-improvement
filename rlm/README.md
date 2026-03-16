# RLM — Recursive Language Model

Drop-in replacement for `llm.completion()` that handles unbounded context via recursive REPL-based code execution. The LLM never sees the full context — it writes Python code that queries, slices, and aggregates a context variable stored in a REPL environment.

**Key result: processes 10M+ tokens at inference with zero degradation. RLM(small model) outperforms standard(large model) by 2x.**

## Quick Start

```bash
pip install -e .
make test    # 210 tests, 98% coverage

# Single query
python scripts/run_query.py --query "What is the revenue?" --context data.txt --verbose

# Benchmarks
python scripts/run_benchmark.py --benchmark oolong --context-scaling --report
```

## Usage

```python
import rlm

result = rlm.completion(
    prompt="Summarize the key findings",
    model="gpt-4o-mini",
    context=open("huge_document.txt").read()  # 10M+ tokens OK
)
print(result.value)
```

## How It Works

The LLM never receives the full context. Instead:
1. Context is loaded as a variable `CONTEXT` in a REPL
2. LLM writes code (peek, grep, search, chunk) to interact with it
3. For large contexts, LLM spawns recursive sub-LMs via `rlm_sub_query()`
4. Results are signaled via `FINAL("answer")` or `FINAL_VAR("variable")`

## Emergent Strategies

| Strategy | When | Pattern |
|----------|------|---------|
| **Peek-then-grep** | Targeted queries | Peek start, grep keywords, read relevant sections |
| **Map-reduce** | Aggregation queries | Chunk context, sub-query per chunk, aggregate |
| **Hierarchical** | Deep analysis | Recursively summarize sections, then summarize summaries |
| **Iterative refinement** | Exploratory queries | Broad grep, narrow down, specific grep |
| **Direct** | Small context | Process in one shot |

## License

MIT
