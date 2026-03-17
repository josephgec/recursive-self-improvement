# Reward Hacking Mitigations + Safety Deliverables

Three defense mechanisms against reward hacking plus per-phase-gate safety packaging (S.1-S.4 consolidation).

## Quick Start

```bash
pip install -e .
make test    # 150 tests, 94% coverage
python scripts/run_eppo_training.py --config configs/eppo.yaml
python scripts/check_reward_hacking.py
python scripts/package_safety.py --phase 2a --iterations 0-15
```

## Three Defense Mechanisms

| Mechanism | Prevents | How |
|-----------|----------|-----|
| EPPO | Policy collapse | Entropy bonus in PPO loss |
| Clip + Delta | Reward spikes | Bound values and deltas |
| Energy monitoring | Representation homogenization | Track activation norms |

## Safety Package (S.1-S.4)

Every phase gate requires a valid safety package with all four tracks green:
- S.1 GDI: drift trajectory and alerts
- S.2 Constraints: pass/fail log and audit chain
- S.3 Interpretability: activation diffs and anomalies
- S.4 Reward hacking: entropy, bounding, energy, shortcuts

## License

MIT
