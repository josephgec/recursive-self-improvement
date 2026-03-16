# SOAR Hindsight Learning

Converts evolutionary search trajectories into fine-tuning data, closing the SOAR self-improvement loop: better search → richer trajectories → better model → better search.

## Quick Start

```bash
pip install -e .
make test                # 255 tests, 98% coverage
python scripts/collect_trajectories.py --results-dir ../soar-evolution/data/results/
python scripts/synthesize_data.py --target-size 5000 --output-format openai
python scripts/run_soar_iteration.py --iterations 3 --report
```

## The Virtuous Cycle

```
Search (generate programs) → Collect trajectories → Synthesize training data →
Fine-tune model → Evaluate improvement → Better search → ...
```

## 6 Training Data Strategies

| Strategy | Source | What it teaches | Weight |
|----------|--------|----------------|--------|
| Direct solution | Solved tasks | Correct ARC solutions | 25% |
| Error correction | Mutation improvements | Debug and fix programs | 25% |
| Improvement chain | Multi-step improvements | Iterative refinement | 20% |
| Hindsight relabel | Failed + partial solutions | Simpler variant solutions | 15% |
| Crossover pairs | Successful crossovers | Combine partial solutions | 10% |
| Pattern description | Solved tasks + LLM | Plan-then-code approach | 5% |

## Architecture

| Module | Purpose |
|--------|---------|
| `src/collection/` | Harvest trajectories, persistent database, indexing |
| `src/synthesis/` | 6 strategies, quality filtering, deduplication, formatting |
| `src/finetuning/` | OpenAI and local fine-tuning, model evaluation, registry |
| `src/iteration/` | Full SOAR loop, improvement tracking, convergence detection |

## License

MIT
