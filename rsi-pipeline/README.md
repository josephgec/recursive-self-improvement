# Composed RSI Pipeline

Wires all four RSI paradigms — SOAR, SymCode/CTM, Gödel Agent, RLM — into a single continuously running self-improvement loop. The system proposes improvements (SOAR), verifies they're correct and compact (SymCode+CTM), applies them to its own code (Gödel), and scales over unbounded context (RLM).

## Quick Start

```bash
pip install -e .
make test    # 215 tests, 95% coverage
python scripts/run_pipeline.py --iterations 10 --report
python scripts/run_single_iteration.py --iteration 1 --verbose
python scripts/run_safety_audit.py --full
```

## The Loop

```
1. SOAR generates candidate improvements to agent code
2. SymCode + CTM verifies candidates (correct AND compact)
3. Gödel Agent applies the best candidate (with rollback)
4. RLM scales everything over unbounded context
5. Safety checks + human review at milestones
6. Hindsight feeds outcomes back to SOAR
```

## Architecture

| Module | Paradigm | Role |
|--------|----------|------|
| `src/outer_loop/` | SOAR | Generate candidate improvements via evolutionary search |
| `src/verification/` | SymCode + CTM | Dual gate: empirical correctness + algorithmic compactness |
| `src/self_modification/` | Gödel Agent | Apply modifications with rollback safety net |
| `src/scaling/` | RLM | Process unbounded context via recursive REPL |
| `src/safety/` | Cross-cutting | GDI, constraints, CAR, emergency stop, human review |

## Safety

- **GDI monitoring** — Goal Drift Index tracks alignment across iterations
- **Hard constraints** — Accuracy floor (80%), entropy floor, drift ceiling
- **CAR tracking** — Capability-Alignment Ratio must be >= 1.0 (Pareto improvement)
- **Emergency stop** — Triggers on CAR < 0.5, 3 consecutive rollbacks, or constraint violation
- **Human checkpoints** — Pause for review every N iterations

## License

MIT
