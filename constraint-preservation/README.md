# Constraint Preservation

Hard binary constraints that gate every modification in the RSI pipeline. No override. No soft fallback. Pass or reject.

## Quick Start

```bash
pip install -e .
make test    # 99 tests, 98% coverage
python scripts/run_check.py --agent-config agent.yaml
python scripts/run_headroom.py
```

## 7 Core Constraints

| Constraint | Category | Threshold | Tolerance |
|-----------|----------|-----------|-----------|
| Accuracy floor | Quality | >= 80% | 0% |
| Entropy floor | Diversity | >= 3.5 bits | 0% |
| Safety eval | Safety | 100% pass | 0% |
| Drift ceiling | Alignment | GDI <= 0.40 | 0% |
| Regression guard | Quality | Max -3pp | 0pp |
| Consistency | Quality | >= 85% | 0% |
| Latency ceiling | Performance | P95 <= 30s | 0s |

All constraints are BINARY, HARD, and IMMUTABLE. The agent cannot modify them.

## Integration

```python
gate = ConstraintGate.from_config("configs/default.yaml")
decision = gate.wrap_modification(
    lambda: agent.apply(candidate),
    agent_state, context
)
# decision.allowed is True or False. No override.
```

## Immutable Audit Log

Every check — pass or fail — is logged with SHA-256 hash chain integrity verification.

## License

MIT
