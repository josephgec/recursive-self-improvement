# RSI Pipeline

Recursive Self-Improvement Pipeline integrating SOAR, SymCode/CTM, Godel Agent, and RLM.

## Quick start

    make test        # run all tests
    make coverage    # run tests with coverage report
    make dry         # dry-run single iteration
    make run         # run full pipeline
    make safety      # run safety audit
    make report      # generate analysis report

## Architecture

The pipeline runs a 6-step iteration loop:
1. **Generate** candidates via StrategyEvolver (SOAR-style)
2. **Verify** candidates through DualVerifier (empirical + compactness gates)
3. **Modify** agent code via ModificationEngine (with safety preconditions)
4. **Evaluate** post-modification performance
5. **Safety** check via GDI monitor, constraint enforcer, CAR tracker
6. **Hindsight** adaptation feeding outcomes back to SOAR

## Key constraints

- All tests are offline with mocks; no external dependencies required.
- Safety gates enforce accuracy floor, entropy floor, drift ceiling.
- Emergency stop triggers on CAR < 0.5, 3 consecutive rollbacks, or constraint violation.
