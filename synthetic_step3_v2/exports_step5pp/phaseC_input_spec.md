# Phase C Input Spec

## 1. Frozen Version

- Freeze timestamp: `2026-03-18T20:54:14.5719628+08:00`
- Benchmark version: `phaseA_v0.1_synth`
- Dataset for Phase C round 1: `synthetic`
- Canonical artifacts root: `C:/Users/cyl/Desktop/phaseC_artifacts`
- Split artifact: `phaseC_round1_split.json`
- Training-config artifact: `phaseC_round1_train_config.json`
- Scope of this freeze:
  - freeze the signal inputs carried from Phase A/Phase B
  - freeze the first-round evaluation protocol
  - do not change any `iTransformer` files in this step

This freeze is based on synthetic signal-side evidence only. It does not claim that the same threshold rules will remain valid on real datasets.

## 2. Phase C Input Signals

### Gating Lambda

- Variant: `score_gating`
- Window / k: `40 / 2`
- Hash: `f4bc9d33f862bbd93db5af7ae20b4edb995cae98`
- Source CSV: `synthetic_step3_v2/exports_step4/rescored_results_20260311_162910.csv`
- Canonical artifact: `lambda_gating_locked.npy`
- Lock status: frozen by Phase B and no longer follows the candidate pool

### Regime Lambda

- Variant: `score_regime`
- Accepted baseline window / k: `50 / 2`
- Hash: `6db132725d81c7889924901173a9399221293406`
- Source CSV: `synthetic_step3_v2/exports_step4/rescored_results_20260306_142434.csv`
- Canonical artifact: `lambda_regime_baseline.npy`
- Status: accepted Phase B baseline
- Artifact provenance: reconstructed from historical `step5pp_utils.py` at commit `6bcfe93` to preserve the accepted-baseline hash exactly

Current accepted baseline metrics:

- `switch_band_correct_rate = 0.83`
- `switch_margin_gap_signed = 0.10248515754938126`
- `peak_delay_min = 121`
- `directional_align_overall = 0.5444463115442782`

## 3. Frozen Phase B Conclusion

Phase B established the following conclusion under the frozen synthetic benchmark:

- `gating` is stable enough to be used as a fixed input signal.
- `regime` remains meaningful, but the accepted baseline is still `50,2`.
- Faster or more separable candidates were found, but they were not promoted because they weakened the defining regime metrics.

Notable rejected candidates:

1. `80,2`
   - Faster and more separable than the accepted baseline.
   - Rejected because `switch_band` and `switch_margin_gap` were weaker.
   - `iteration_accept = false`
   - `iteration_accept_reason = mechanism_progress_lt_2of3`
2. `120,8`
   - Much faster and more separable than the accepted baseline.
   - Rejected because `switch_band` and `switch_margin_gap` collapsed.
   - `iteration_accept = false`
   - `iteration_accept_reason = pass_core_checks_v3_v2_false;mechanism_progress_lt_2of3`

Implication for Phase C:

- Use `40,2` for `gating`.
- Use accepted baseline `50,2` for `regime`.
- Do not silently replace the regime input with a rejected candidate.

## 4. Phase C Round-1 Protocol

### Arms

Run the first model-side experiment with exactly four arms:

1. `baseline`
2. `gating-only`
3. `regime-only`
4. `gating+regime`

### Integration Method

Keep the first round minimal and weakly invasive.

For `gating`:

- use loss weighting, sample weighting, or timestep weighting
- do not directly control a complex structural module in round 1

For `regime`:

- use an extra time feature or a light auxiliary input channel
- do not directly implement large-scale structural switching in round 1

For `gating+regime`:

- combine the two minimal integrations only after the single-signal runs are defined under the same protocol

### Evaluation Metrics

Fix the evaluation metric families before training:

- `global` metrics
- `switch-window` metrics
- `pre/post` metrics
- `high/low-lambda-subset` metrics

Do not revise these metric families after the runs start.

### Seed, Split, and Training Budget

Freeze the following round-1 experiment constraints before any model-side coding:

- fixed seed: `2026`
- one seed only in round 1
- one shared split for all four arms
- no per-arm re-splitting
- one shared training budget for all four arms
- no per-arm hyperparameter tuning in round 1

Operational meaning:

- `baseline`, `gating-only`, `regime-only`, and `gating+regime` must run on the same split.
- the evaluation slices must preserve both pre and post regions
- the switch-window evaluation slice must include `t_switch = 3600` with enough local context for the fixed switch-window metrics
- optimizer family, epoch budget, early-stop rule, batch size, and learning-rate setting must stay identical across the four arms
- the concrete shared split is now frozen in `phaseC_round1_split.json`

This document freezes the protocol constraints now. The concrete split artifact is already frozen as `phaseC_round1_split.json`. The concrete model training config should be created next, before any `iTransformer` code is modified.

## 5. Concrete Training Config Freeze

The first model-side training config is now frozen in `phaseC_round1_train_config.json`.

What this artifact already fixes against the real `iTransformer` code:

- training entrypoint: `run.py`
- experiment class: `Exp_Long_Term_Forecast`
- data provider: `data_provider.data_factory.data_provider`
- optimizer family: `Adam`
- loss: `MSE`
- learning rate: `1e-4`
- lr schedule: `type1`
- epoch budget: `10`
- batch size: `32`
- patience: `3`
- AMP: `False`
- round-1 worker count: `0`

What this artifact explicitly does **not** pretend is solved:

- upstream `run.py` still hardcodes `fix_seed = 2023`, while Phase C round 1 is frozen to `seed = 2026`
- upstream `Dataset_Custom` still hardcodes a `70/10/20` split and ignores `phaseC_round1_split.json`
- upstream `Dataset_Custom` still expects a CSV with a `date` column, while the current synthetic source is `X.npy`

So the next model-side implementation step is not "start training immediately". It is:

- add a split-aware synthetic dataset adapter
- add a seed override path so model-side runs can actually use `2026`
- keep all four arms on the same split and frozen training budget

## 6. Canonical Artifact Ownership

- `C:/Users/cyl/Desktop/phaseC_artifacts` is the Phase C canonical master.
- The synthetic project is responsible for writing and updating this directory.
- The `iTransformer` project is responsible for reading from this directory and must not write back into it.
- If a copy is later placed inside the model repository, it is only a snapshot, not the master.

## 7. Phase C Round-1 Success Criteria

Use the following interpretation for the first round:

1. If `gating-only` beats `baseline` on global metrics, then the gating signal has model-side value.
2. If `regime-only` beats `baseline` on switch-window metrics, then the regime signal has model-side value.
3. If `gating+regime` is best overall or shows complementary gains without harming the switch-window metrics, then the two signals are complementary.

These criteria are meant to decide whether the signal-side work from Phase A/Phase B transfers into model-side value. They are not meant to re-open the synthetic benchmark itself.

## Current Model-side Status

- Phase C protocol-alignment milestone is complete.
- `Dataset_PhaseC_Synthetic` is implemented and validated against `phaseC_round1_split.json`.
- `run.py` now exposes `--seed`; round-1 baseline was validated under `seed=2026`.
- A no-op gating control using `lambda_gating_locked.npy` runs and matches baseline exactly.
- Next step: real `gating-only` with minimal `loss weighting`.
