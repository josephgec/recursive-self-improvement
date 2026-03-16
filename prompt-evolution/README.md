# Reasoning-Guided Prompt Evolution

Genetic algorithm that evolves system prompts using a thinking model (reasoning-capable LLM) as the evolutionary operator. Produces prompts measurably better than human-engineered ones.

**Target: ~11% fitness improvement over non-thinking baselines on financial mathematical reasoning.**

## Quick Start

```bash
pip install -e .
make test                # 202 tests, 95% coverage
python scripts/run_evolution.py --domain financial_math --generations 20 --report
python scripts/run_ablation.py --conditions all --repetitions 3 --report
```

## How It Works

System prompts are structured as **genomes** with typed sections (persona, reasoning instructions, domain knowledge, constraints, output format). The GA evolves these sections using a thinking model that reasons about why prompts fail and how to fix them.

```
Initialize diverse prompts → [Evaluate → Select → Mutate/Crossover → Evaluate] × N → Best prompt
```

## 7 Ablation Conditions

| Condition | Init | Mutate | Crossover | Evaluate | Tests |
|-----------|------|--------|-----------|----------|-------|
| Thinking all | Think | Think | Think | Think | Full system |
| Think mutate only | Simple | Think | Simple | Simple | Mutation matters most? |
| Think eval only | Simple | Simple | Simple | Think | Evaluation matters most? |
| Think init only | Think | Simple | Simple | Simple | Starting population matters? |
| Non-thinking | Simple | Simple | Simple | Simple | Baseline |
| Human engineered | N/A | N/A | N/A | Simple | Expert prompt |
| Random search | Random | N/A | N/A | Simple | Lower bound |

## Financial Math Benchmark

8 categories with programmatically generated tasks (correct answers computed via exact formulas):

- Compound interest, present value, loan amortization
- Option pricing, risk & return, bond valuation
- Tax optimization, time value (NPV/IRR)

## Architecture

| Module | Purpose |
|--------|---------|
| `src/genome/` | Structured prompt genome with typed sections |
| `src/operators/` | Thinking + non-thinking init, mutation, crossover, evaluation |
| `src/ga/` | GA engine with elitism, diversity maintenance, stagnation detection |
| `src/evaluation/` | Financial math benchmark with answer checking |
| `src/comparison/` | 7-condition ablation with statistical testing |
| `src/deliverables/` | Phase 2a deliverable packaging |

## License

MIT
