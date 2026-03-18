# Phase C Artifacts

This directory is the canonical master for Phase C round-1 artifacts.

Ownership:
- synthetic project: writes and updates this directory
- iTransformer project: reads from this directory only
- copies inside any project repo are snapshots, not the master

Core files:
- `phaseC_freeze.json`: single source of truth for the frozen Phase C round-1 inputs and protocol
- `phaseC_input_spec.md`: human-readable explanation of the freeze and protocol
- `phaseC_round1_split.json`: shared split artifact for all four round-1 arms
- `lambda_gating_locked.npy/.npz`: locked Phase B gating signal
- `lambda_regime_baseline.npy/.npz`: accepted Phase B regime baseline signal

Path rule:
- paths inside `phaseC_freeze.json` are resolved relative to `canonical_artifacts_root`
- do not maintain a second independent "latest" copy elsewhere
