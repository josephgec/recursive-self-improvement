# Gödel Agent

Self-referential agent that modifies its own meta-learning algorithm. After each task batch, the agent inspects its runtime state, deliberates on whether its learning strategy is effective, and can rewrite its own code — including the code that decides *what* to rewrite.

## What It Does

1. **Meta-learning core** — Modifiable algorithm (prompt templates, few-shot selectors, reasoning strategies) that the agent inspects and rewrites at runtime via AST manipulation and monkey-patching.
2. **Rollback-guarded modification** — Every self-modification is validated against an immutable test suite before committing; degradation triggers automatic revert.
3. **Full-diff audit log** — Every modification is recorded with code diffs, performance deltas, and the agent's reasoning trace.

## Quick Start

```bash
# Install
pip install -e .

# Run tests (290 tests, ~1 sec)
make test

# Debug run (5 iterations, mock LLM)
python scripts/run_agent.py --config configs/debug.yaml --tasks math

# Full math specialist run
python scripts/run_agent.py --config configs/math_specialist.yaml --tasks math --iterations 50

# Static vs. self-modifying comparison
python scripts/run_ablation.py

# Browse modification audit log
python scripts/inspect_history.py data/audit_logs/latest/
```

## Architecture

```
src/
├── core/           # Agent loop, state management, runtime inspector
├── meta/           # Modifiable meta-learning algorithm
├── modification/   # Code inspector, modifier, deliberation engine
├── validation/     # Immutable validation suite, rollback mechanism
├── audit/          # Diff engine, modification logger, safety hooks
├── tasks/          # Task domains (math, code, science)
└── analysis/       # Modification history, complexity tracking, reports
```

## Key Invariant

The agent **can** modify: prompt strategy, few-shot selector, reasoning strategy.

The agent **cannot** modify: validation suite, rollback mechanism, audit logger, safety hooks.

This is enforced at the code level via target whitelists, not by prompting.

## Self-Modification Flow

```
Performance drops -> Deliberation triggered -> LLM proposes code change ->
Risk assessment -> Checkpoint -> Apply modification -> Validate against
immutable test suite -> Accept (update baseline) or Rollback (restore state) ->
Log full diff to audit trail
```

## What the Agent Actually Modifies

**Iteration 0 (initial):** `reasoning_strategy.choose()` returns `"cot"` for everything.

**Iteration 8:** Agent modifies `choose()` to use code for math:
```python
def choose(self, task, recent_results):
    if task.category == "math":
        return "code"
    return "cot"
```

**Iteration 14:** Agent adds direct answering for easy tasks:
```python
def choose(self, task, recent_results):
    if task.category == "math":
        return "code"
    if task.difficulty <= 2:
        return "direct"
    return "cot"
```

Each modification is proposed by the agent's LLM, validated, and logged with full diffs.

## Safety Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| **Target whitelist** | Only allowed components can be modified |
| **AST security scanner** | Blocks dangerous imports, exec/eval, dunder access |
| **Immutable validation** | Test suite cannot be modified or viewed by the agent |
| **Automatic rollback** | Failed modifications are reverted to last checkpoint |
| **Complexity bounds** | Alert if agent code grows beyond 5x initial complexity |
| **Modification rate limit** | Prevents thrashing (>3 mods in 5 iterations) |
| **Deliberation depth** | Multi-level "think before acting" before modifying |

## Configuration

```yaml
meta_learning:
  max_iterations: 50
  warmup_iterations: 5        # No modifications during warmup
  modification_cooldown: 2    # Min iterations between modifications

modification:
  require_deliberation: true
  deliberation_depth: 2       # Levels of meta-reasoning
  allowed_targets: ["prompt_strategy", "few_shot_selector", "reasoning_strategy"]

validation:
  min_pass_rate: 0.90         # Must pass 90% of validation tasks
  performance_threshold: -0.05  # Max 5% degradation allowed
  auto_rollback: true
```

## Testing

```bash
# All tests (290 tests, 92% coverage)
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## License

MIT
