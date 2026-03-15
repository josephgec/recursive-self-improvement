# Godel Agent Fragility Testing

Adversarial stress testing of the self-modifying Godel Agent.

## Quick start
- `make install`
- `python scripts/run_stress_test.py --config configs/quick.yaml --report` -- quick run
- `python scripts/run_stress_test.py --config configs/full.yaml --complexity-sweep --report` -- full
- `python scripts/run_single_scenario.py --scenario "syntax_error_injection"` -- one scenario
- `python scripts/run_complexity_sweep.py` -- find complexity ceiling

## Architecture
- `src/adversarial/` -- Adversarial scenarios (fault injection, self-reference attacks, complexity escalation)
- `src/measurement/` -- Recovery tracking, failure classification, comprehension probes
- `src/harness/` -- Stress test runner with isolated environments
- `src/analysis/` -- Failure landscape, complexity ceiling, fragility score, report

## Key question
At what cumulative complexity does the LLM lose the ability to comprehend its own code?
