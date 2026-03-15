# RSI Infrastructure

Shared infrastructure for all RSI experiment phases: sandboxed REPL,
symbolic math execution, and experiment tracking with safety metrics.

## Quick start
- `make install` -- install dependencies
- `make test` -- run tests
- `python scripts/run_smoke_test.py` -- validate everything works

## Architecture
- `repl/` -- Sandboxed Python REPL (local, Docker, Modal backends)
- `symbolic/` -- SymPy + Z3 execution sandbox
- `tracking/` -- W&B + safety metrics (Goal Drift Index, constraints, CAR)
- `sdk/` -- Unified client API for downstream phases

## Usage from downstream phases
```python
from sdk import REPLClient, SymbolicClient, TrackingClient, InfraConfig

config = InfraConfig.from_yaml("configs/local.yaml")

# REPL
async with REPLClient.from_config(config).session() as repl:
    result = repl.execute("x = 42")

# Symbolic verification
client = SymbolicClient.from_config(config)
result = client.verify_code(llm_code, expected="[-2, 2]")

# Tracking + safety
tracker = TrackingClient.from_config(config)
tracker.start_run("my_experiment")
tracker.log_generation(0, metrics)
tracker.check_safety(0, model, eval_texts)
```

## Key config
- `repl.backend`: "local" | "docker" | "modal"
- `symbolic.backend`: "subprocess" | "docker"
- `tracking.backend`: "wandb" | "local"
