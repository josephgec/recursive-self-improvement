# Success Criteria Evaluation

Month-18 GO/NO-GO verdict on the RSI research program. 5 pre-registered criteria, each with binary pass/fail and statistical confidence.

## Quick Start

```bash
pip install -e .
make test    # 119 tests, 97% coverage
python scripts/run_evaluation.py --verify-integrity --report --package
```

## 5 Criteria

| # | Criterion | Threshold | Evidence |
|---|-----------|-----------|---------|
| 1 | Sustained improvement | ≥10 iters, ≥5pp gain, ≥10pp above collapse | Phase 3 curves |
| 2 | Paradigm improvement | Each of 4 paradigms beats baseline (p<0.05) | Phase 4 ablations |
| 3 | GDI within bounds | Max ≤0.50, ≤5 consecutive yellow | S.1 data |
| 4 | Publications | ≥2 accepted, ≥1 tier-1/2 | Publication records |
| 5 | Auditability | All logs complete, chain intact, ≥20 traces | Audit trail |

## Verdict

- **SUCCESS**: 5/5 criteria met
- **PARTIAL**: 3-4/5 criteria met
- **NOT MET**: ≤2/5 criteria met

## License

MIT
