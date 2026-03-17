# Key Risks and Mitigations

Operational risk management across 6 domains with detection, prevention, and automated response protocols.

## Quick Start

```bash
pip install -e .
make test    # 288 tests, 98% coverage
python scripts/run_risk_check.py
python scripts/check_budget.py
python scripts/check_deadlines.py
```

## 6 Risk Domains

| # | Risk | Detection | Prevention | Response |
|---|------|-----------|-----------|----------|
| 1 | Collapse | Forecaster (template matching) | Conservative α, data reserve | Halt-and-diagnose |
| 2 | Self-mod | Staging, blast radius | Complexity budget, quarantine | Rollback |
| 3 | Reward hacking | Eval gap, audit, shortcuts | EPPO, eval rotation | Stop training |
| 4 | Cost explosion | Budget manager, burn rate | Hard caps, circuit breakers | Kill queries |
| 5 | Over-conservative | Tightness detector | Graduated relaxation | Loosen + compensate |
| 6 | Publication | Deadline tracker, readiness | Early writing, auto-generation | Workshop fallback |

## License

MIT
