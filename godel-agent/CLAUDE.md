# Godel Agent

Self-referential agent that modifies its own meta-learning algorithm.

## Quick start
- `make install` -- install deps
- `make test` -- run tests
- `python scripts/run_agent.py --config configs/debug.yaml --tasks math` -- debug run
- `python scripts/run_ablation.py` -- static vs self-modifying comparison

## Architecture
- `src/core/` -- Agent loop, state management, runtime inspector
- `src/meta/` -- Modifiable meta-learning algorithm (prompt, few-shot, reasoning)
- `src/modification/` -- Code inspector, modifier, deliberation engine
- `src/validation/` -- Immutable validation suite, rollback mechanism
- `src/audit/` -- Diff engine, modification logger, safety hooks

## Key invariant
The agent can modify: prompt strategy, few-shot selector, reasoning strategy.
The agent CANNOT modify: validation suite, rollback mechanism, audit logger.

## Self-modification flow
Performance drops -> deliberation triggered -> LLM proposes code change ->
risk assessment -> checkpoint -> apply -> validate -> accept or rollback -> log diff
