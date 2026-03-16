# Constraint Preservation

Hard binary constraint system that gates every RSI pipeline modification.

## Quick start

```bash
make test        # run all tests
make coverage    # run tests with coverage (target >= 90%)
make check       # run constraint check against mock agent
make headroom    # compute headroom report
make report      # generate full analysis report
```

## Architecture

- `src/constraints/` -- individual constraint implementations (accuracy, entropy, safety, drift, regression, consistency, latency, custom)
- `src/checker/` -- constraint suite assembly, parallel runner, verdict, caching
- `src/enforcement/` -- gate (no override), rejection handler, rollback trigger, SHA-256 audit log
- `src/evaluation/` -- held-out tasks, safety prompts, diversity probes, regression benchmarks
- `src/monitoring/` -- headroom monitor, trend detector, dashboard config
- `src/analysis/` -- rejection analysis, constraint tightness, report generation

## Key invariants

- All constraints are immutable once registered.
- Safety constraint has zero tolerance (100% pass rate required).
- Audit log is append-only with SHA-256 hash chain for tamper detection.
- Gate decisions cannot be overridden.
- All evaluation is offline and deterministic.
