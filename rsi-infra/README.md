# RSI Infrastructure

Shared infrastructure layer for all Recursive Self-Improvement experiment phases. Provides three core systems:

1. **Sandboxed REPL** — Isolated Python execution environments with memory persistence across calls, supporting recursive sub-REPL spawning for the RLM architecture.
2. **Symbolic Math Execution** — SymPy + Z3 sandboxes for deterministic verification of LLM-generated mathematical solutions.
3. **Experiment Tracking with Safety Metrics** — Goal Drift Index, constraint preservation, Capability-Alignment Ratio, and alerting.

## Quick Start

```bash
# Install
pip install -e .

# Run tests (296 tests, ~21 sec)
make test

# Smoke test (validates all systems)
python scripts/run_smoke_test.py
```

## Architecture

```
rsi-infra/
├── repl/           # Sandboxed Python REPL (local, Docker, Modal backends)
├── symbolic/       # SymPy + Z3 execution sandbox with verification harness
├── tracking/       # Experiment tracking + safety metrics
└── sdk/            # Unified client API for downstream phases
```

## REPL Sandbox

Isolated Python execution with variable persistence across calls and recursive child spawning.

```python
from sdk import REPLClient, InfraConfig

config = InfraConfig.from_yaml("configs/local.yaml")
client = REPLClient.from_config(config)

# Execute code with persistent state
repl = client.create_session_sync()
result = repl.execute("x = 42")
result = repl.execute("print(x * 2)")  # prints 84

# Spawn isolated child (inherits variables)
child = repl.spawn_child()
child.execute("x = 100")  # doesn't affect parent
assert repl.get_variable("x") == 42
```

**Backends:** Local (in-process, for dev), Docker (isolated containers), Modal (cloud scale with GPU).

**Security:** AST-level code analysis blocks dangerous imports (`os`, `subprocess`, `socket`), dunder access (`__class__`, `__subclasses__`), and unsafe builtins (`exec`, `eval`). Container-level isolation (network disabled, read-only filesystem) provides defense in depth.

## Symbolic Math Execution

Execute SymPy and Z3 code with structured results and solution verification.

```python
from sdk import SymbolicClient, InfraConfig

config = InfraConfig.from_yaml("configs/local.yaml")
client = SymbolicClient.from_config(config)

# Solve equations
result = client.solve("x**2 - 4", variable="x")
# result.expression = "[-2, 2]"

# Verify LLM-generated code
verified = client.verify_code(
    "from sympy import *; answer = solve(x**2 - 4, x)",
    expected="[-2, 2]"
)

# Check logical implications
assert client.check_implication(["x > 5"], "x > 3")
```

## Experiment Tracking + Safety

Track metrics with built-in safety monitoring: Goal Drift Index, constraint preservation, and alerting.

```python
from sdk import TrackingClient, InfraConfig

config = InfraConfig.from_yaml("configs/local.yaml")
tracker = TrackingClient.from_config(config)
tracker.start_run("my_experiment", {"model": "llama-7b"})

for gen in range(15):
    metrics = {"train_loss": 0.5, "eval_accuracy": 0.9}
    tracker.log_generation(gen, metrics)

    # Safety check: GDI + constraints + alerts in one call
    safety = tracker.check_safety(
        generation=gen,
        generated_texts=["sample output..."],
        metrics=metrics
    )
    if safety.recommendation == "halt":
        break

tracker.finish()
```

### Safety Metrics

| Metric | Description |
|--------|-------------|
| **Goal Drift Index** | Composite of semantic, lexical, structural, and distributional drift from reference |
| **Constraint Preservation** | Hard checks (accuracy floor, safety eval, entropy floor) with revert/halt actions |
| **Capability-Alignment Ratio** | capability_gain / alignment_cost — tracks whether improvements come at alignment cost |

### Backends

- **W&B** — Full experiment tracking with custom safety dashboards
- **Local** — File-based JSONL tracking (no external service needed)

## Configuration

```yaml
# configs/default.yaml
repl:
  backend: "local"          # "local" | "docker" | "modal"
  timeout_seconds: 300
  max_memory_mb: 4096
  max_recursion_depth: 10

symbolic:
  backend: "subprocess"     # "subprocess" | "docker"
  sympy_timeout: 60
  z3_timeout: 120

tracking:
  backend: "local"          # "wandb" | "local"
  safety:
    alert_threshold_drift_cosine: 0.15
    constraint_preservation_mode: "hard"
```

## Downstream Phase Consumption

| Phase | REPL | Symbolic | Tracking |
|-------|------|----------|----------|
| Model Collapse Baselines | — | — | Collapse metrics, GDI |
| SymCode (Neurosymbolic) | Execute LLM code | Verify via SymPy + Z3 | Verification pass rates |
| Godel Agent | Self-modifying code | Verify modifications | GDI after each modification |
| SOAR | Execute candidate programs | — | Evolutionary fitness, GDI |
| RLM | Recursive REPL execution | — | Test-time scaling |

## Testing

```bash
# Unit tests (296 tests, 94% coverage)
pytest -v --tb=short

# With coverage report
pytest --cov=repl/src --cov=symbolic/src --cov=tracking/src --cov=sdk --cov-report=term-missing

# Smoke test (validates all systems end-to-end)
python scripts/run_smoke_test.py
```

## License

MIT
