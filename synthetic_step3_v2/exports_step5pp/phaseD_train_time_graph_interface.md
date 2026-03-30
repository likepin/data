# Stage 1 Freeze And Interface

Date: `2026-03-29`

## Purpose

This note freezes what already exists in the synthetic pipeline and translates it
into a model-facing interface for the new mainline:

`single canonical lambda + PCMCI A_base + window-level DeltaA + train-time soft attention bias`

This is a Stage 1 interface note only. It does not change model code.

The key semantic rule for this interface is:
- `lambda` remains an external clustering-derived signal
- `lambda` is not derived from `A_base` or `DeltaA`
- `lambda^(w)` is obtained by aggregating a canonical timeline-level `lambda_t` over each model window
- higher `lambda^(w)` means stronger fallback to the conservative static causal skeleton
- lower `lambda^(w)` means stronger trust in local dynamic correction

## Already Frozen And Reusable

### 1. Split Contract

Canonical source:
- `C:/Users/cyl/Desktop/phaseC_artifacts/phaseC_round1_split.json`

Frozen facts:
- total length: `6000`
- indexing: zero-based, half-open intervals
- `t_switch = 3600`
- train intervals: `[0, 2400)`, `[3800, 5000)`
- val intervals: `[2400, 2900)`, `[5000, 5500)`
- test intervals: `[2900, 3400)`, `[5500, 6000)`

This remains the canonical split contract for any synthetic-to-model interface.

### 2. Training Window Contract

Canonical source:
- `C:/Users/cyl/Desktop/phaseC_artifacts/phaseC_round1_train_config.json`

Frozen facts from `frozen_round1_training_config`:
- `seq_len = 96`
- `label_len = 48`
- `pred_len = 96`
- `enc_in = dec_in = c_out = 10`
- `features = M`
- `batch_size = 32`
- `train_epochs = 10`
- `patience = 3`
- `learning_rate = 1e-4`

Validated synthetic dataset smoke check:
- train windows: `3218`
- val windows: `618`
- test windows: `618`

These values define the future window-index contract.

### 3. Existing Lambda Artifacts

Canonical Phase C artifacts:
- `C:/Users/cyl/Desktop/phaseC_artifacts/lambda_gating_locked.npz`
- `C:/Users/cyl/Desktop/phaseC_artifacts/lambda_regime_baseline.npz`

Observed facts:
- both arrays have shape `(6000,)`
- `gating` valid count: `5961`, first valid index: `39`
- `regime` valid count: `5951`, first valid index: `49`
- both are timeline-level sequences, not window-level tensors

Current status for the new mainline:
- these two artifacts are the current external clustering-derived lambda candidates
- the canonical timeline-level `lambda_t` is fixed to `lambda_gating_locked`
- before any window aggregation, `lambda_t` must be sanitized exactly as in the Phase C dataset loader: linear interpolation over internal NaNs with edge-value extrapolation at the prefix/suffix
- the model-facing `lambda^(w)` is obtained by taking the mean of the sanitized `lambda_t` over the encoder-history interval of each training window

### 4. Existing Graph Artifacts

Current synthetic graph files:
- `synthetic_step3_v2/A_base.npy`
- `synthetic_step3_v2/adj_base.npy`
- `synthetic_step3_v2/DeltaA.npy`

Observed shapes:
- `A_base.npy`: `(2, 10, 10)`
- `adj_base.npy`: `(10, 10)`
- `DeltaA.npy`: `(2, 10, 10)`

Interpretation:
- these are global synthetic graph objects
- `A_base.npy` and `DeltaA.npy` come from the synthetic generator / GT structure path
- they are not exported as per-window `A_local^(w)` or `DeltaA^(w)` artifacts

For the new mainline, they can serve as:
- static reference material
- support reference material
- semantic anchor for what the graph objects mean

They do **not** yet satisfy the final model interface on their own.

## What The New Mainline Needs

The model-side mainline needs the following objects:

1. `A_base`
- one static graph prior
- fixed across all training windows

2. `DeltaA^(w)`
- one dynamic graph-drift matrix per training window

3. `lambda^(w)`
- one scalar per training window
- derived from the chosen clustering-based timeline signal `lambda_t`
- obtained by aggregating `lambda_t` on the time interval covered by window `w`
- used as a conservative trust gate on `DeltaA^(w)`, not as a trigger to amplify it

4. `window_index`
- a stable mapping from model dataset window id to source time interval

## Interface Gap

What already exists:
- split contract
- training-window geometry
- historical lambda sequences
- global graph objects

What is still missing:
- a training-window index artifact
- per-window `A_local^(w)` or `DeltaA^(w)` artifacts
- per-window scalar `lambda^(w)` artifacts

## Proposed Model-Facing Contract

The new mainline should read the following four artifact groups:

### A. Immutable Frozen Inputs

- split:
  `C:/Users/cyl/Desktop/phaseC_artifacts/phaseC_round1_split.json`
- training config:
  `C:/Users/cyl/Desktop/phaseC_artifacts/phaseC_round1_train_config.json`
- static graph:
  `synthetic_step3_v2/A_base.npy`
- static support:
  `synthetic_step3_v2/adj_base.npy`

### B. New Derived Window Artifacts

These do not exist yet and should be derived next:

- `phaseD_window_index_train.json`
  - maps each training window id to the exact dataloader-aligned source interval
  - required fields:
    - `sample_id`
    - `interval_id`
    - `interval_local_index`
    - `window_start`
    - `s_begin`
    - `s_end`
    - `r_begin`
    - `r_end`
    - `lambda_start`
    - `lambda_end`
- `phaseD_deltaA_train.npy`
  - shape proposal: `(num_train_windows, 10, 10)`
  - model-facing semantic contract:
    - support-masked
    - lag-aggregated from the local lag-wise ridge coefficients
    - signed drift matrix, not absolute-value-only
    - no extra model-side normalization baked in beyond the fixed export rule
- `phaseD_lambda_train.npy`
  - shape proposal: `(num_train_windows,)`
  - produced by:
    1. sanitizing `lambda_gating_locked` by linear interpolation with edge-value extrapolation
    2. averaging the sanitized timeline signal over the encoder-history interval `[s_begin, s_end)` of each train window
- `phaseD_interface_manifest.json`
  - records source paths, hashes, shape contract, and derivation settings
  - must also record:
    - split hash
    - train-config hash
    - sample-order or `window_starts` hash
    - lambda sanitization rule
    - lambda aggregation rule
    - ridge alpha
    - support-mask source
    - lag aggregation rule
    - signed/unsigned export semantics

## Current Stage 1 Decisions

Frozen now:
- reuse the existing split artifact
- reuse the existing training-window geometry
- reuse the existing global graph files as static references
- fix canonical `lambda_t` to `lambda_gating_locked`
- fix `lambda_t -> lambda^(w)` to: sanitize timeline lambda first, then apply encoder-history mean aggregation
- fix `A_local^(w)` estimation to windowed ridge regression on `PCMCI` parents/support
- fix `window_index` to dataloader-native interval fields based on `window_starts[index] -> s_begin -> s_end -> r_begin -> r_end`
- fix model-facing `DeltaA^(w)` export to support-masked, lag-aggregated, signed `(10, 10)` matrices
- do not modify `phaseC_artifacts`

Still open:
 - whether an additional audit-only lag-wise raw export should be kept alongside the model-facing aggregated export

## Immediate Next Step

Before any model edit:

1. define the exact window index contract from the frozen split and training geometry
2. export `A_local^(w)` by running windowed ridge regression on the fixed `PCMCI` parents/support
3. derive `DeltaA^(w)` from the exported local window graphs
4. export the new window artifacts in one stable manifest-backed bundle

## Mainline Semantics

For the current mainline, the intended relation is:

`A_eff^(w) = A_base + (1 - lambda^(w)) * DeltaA^(w)`

with:
- `A_base`: static graph prior
- `DeltaA^(w)`: dynamic graph correction for window `w`
- `lambda^(w)`: external window-level control scalar obtained from clustering-derived `lambda_t`

`lambda^(w)` is therefore a conservative control signal:
- high `lambda^(w)`: trust `A_base` more and suppress `DeltaA^(w)`
- low `lambda^(w)`: allow `DeltaA^(w)` to contribute more strongly

`lambda^(w)` is not a readout computed from `DeltaA^(w)` itself.

## Local Graph Estimation Rule

The first mainline version fixes:
- run `PCMCI` once on the full training split to define the static parents/support
- do **not** rerun `PCMCI` on every window
- for each train window `w`, estimate `A_local^(w)` by windowed ridge regression only on the fixed `PCMCI` parents/support
- aggregate lag-wise local coefficients into the model-facing window graph
- then define `DeltaA^(w) = A_local^(w) - A_base`

At that point, the model repo can consume the interface without reopening synthetic definitions.
