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

- `window_index_train.json`
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
- `deltaA_train.npy`
  - shape proposal: `(num_train_windows, 10, 10)`
  - model-facing semantic contract:
    - support-masked
    - lag-aggregated from the local lag-wise ridge coefficients
    - signed drift matrix, not absolute-value-only
    - no extra model-side normalization baked in beyond the fixed export rule
- `lambda_train.npy`
  - shape proposal: `(num_train_windows,)`
  - produced by:
    1. sanitizing `lambda_gating_locked` by linear interpolation with edge-value extrapolation
    2. averaging the sanitized timeline signal over the encoder-history interval `[s_begin, s_end)` of each train window
- `interface_manifest.json`
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

Canonical generic export locations:
- synthetic GT backend:
  - `synthetic_step3_v2/exports_step5pp/graph_interface/a_base_agg.npy`
  - `synthetic_step3_v2/exports_step5pp/graph_interface/support.npy`
  - `synthetic_step3_v2/exports_step5pp/graph_interface/window_index_train.json`
  - `synthetic_step3_v2/exports_step5pp/graph_interface/lambda_train.npy`
  - `synthetic_step3_v2/exports_step5pp/graph_interface/deltaA_train.npy`
  - `synthetic_step3_v2/exports_step5pp/graph_interface/interface_manifest.json`
- ETTh1 estimated backend:
  - `interfaces/ETTh1_graph_interface/a_base_agg.npy`
  - `interfaces/ETTh1_graph_interface/support.npy`
  - `interfaces/ETTh1_graph_interface/window_index_train.json`
  - `interfaces/ETTh1_graph_interface/lambda_train.npy`
  - `interfaces/ETTh1_graph_interface/deltaA_train.npy`
  - `interfaces/ETTh1_graph_interface/interface_manifest.json`

Legacy note:
- the older `phaseD_*` filenames under `synthetic_step3_v2/exports_step5pp/phaseD_interface/`
  are kept only for backward compatibility with the already-implemented Stage 3 synthetic sanity path
- all new interface exports should use the generic names above

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

## Stage 2 Status

Completed for the synthetic GT backend:
- lag aggregation is fixed to `sum_over_lags`
- exported graph orientation is fixed to `tgt_src`
- `a_base_agg.npy` has shape `(10, 10)`
- `support.npy` has shape `(10, 10)`
- `lambda_train.npy` has shape `(3218,)`
- `deltaA_train.npy` has shape `(3218, 10, 10)`
- `window_index_train.json` contains `3218` dataloader-aligned train-window records
- `interface_manifest.json` records source hashes, sample-order hash, lambda sanitization, lambda aggregation, and graph export semantics

Observed train-window regime counts:
- pre windows: `2209`
- post windows: `1009`

Completed for the ETTh1 estimated backend:
- export directory: `interfaces/ETTh1_graph_interface/`
- `a_base_agg.npy` has shape `(7, 7)`
- `support.npy` has shape `(7, 7)`
- `lambda_train.npy` has shape `(8449,)`
- `deltaA_train.npy` has shape `(8449, 7, 7)`
- `window_index_train.json` contains `8449` dataloader-aligned train-window records
- `interface_manifest.json` records `PCMCI + ParCorr` settings, ridge alpha, lambda derivation settings, and sample-order hash
- this `ParCorr` bundle is retained as a cheap smoke/debug backend, not as the formal real-data mainline

Canonical exporter:
- `export_graph_interface.py`
  - `synthetic_gt` subcommand: synthetic GT backend with generic naming
  - `real_estimated` subcommand: generic real-data backend with generic naming
  - `etth1_estimated` subcommand: ETTh1 convenience alias for the generic real-data backend

Formal real-data mainline:
- static graph discovery should use nonlinear `PCMCI + CMIknn`
- `ParCorr` remains available only for cheap plumbing checks
- `windowed ridge on fixed support` remains the local `DeltaA^(w)` estimator
- `2026-03-31`: `CMIknn` support was added to the generic exporter and dependencies were resolved
- `2026-04-01`: a full `ETTh1` nonlinear export completed successfully to `interfaces/ETTh1_graph_interface_cmiknn`
- observed runtime for the formal `ETTh1` export was about `590` minutes (`~9.83` hours) with:
  - `cond_test = CMIknn`
  - `tau_max = 2`
  - `pc_alpha = 0.01`
  - `knn = 20`
  - `shuffle_neighbors = 10`
  - `sig_samples = 200`
- the exporter now writes stage logs, `interface_progress.json`, and partial artifacts so future long runs are visible and restart-safe enough for inspection

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
