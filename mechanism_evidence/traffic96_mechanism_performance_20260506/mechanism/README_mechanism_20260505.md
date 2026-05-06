# Traffic96 Log-Tail Quality Guard Mechanism Evidence

Generated: 2026-05-05

Data repo commit: `3b366ea5c71c1a28cf0ee32f3fd0db76885b637c`

This package is a lightweight curated evidence bundle for the Traffic-96 post-hoc calibration case. It copies only small CSV and PNG artifacts. Large array artifacts are referenced in `manifest.json` and `raw_refs/`, but are not copied into this package.

## Scope

This is mechanism evidence, not a headline performance table.

The intended claim is:

CACI can use `log_tail_adaptive` lambda scaling plus rank-based post-hoc calibration to convert Traffic-96 from lambda saturation / guard rejection into a guarded `Selective` mode, while localizing high residual-risk non-stationary segments and exposing fixed-support dynamic graph reconfiguration.

The intended non-claim is:

Traffic-96 does not currently provide a strong final test MSE gain. The final test MSE is effectively neutral/slightly worse, while test MAE improves slightly.

## Key Closed-Loop Result

- Profile: `traffic96_static`
- Selected lambda: `change_slope_no_range`, `window=40`, `k=2`
- Lambda scaling: `log_tail_adaptive`
- Lambda transform: `rank`
- Quality guard: `passed_lambda_quality_guard`
- Quality score: `0.582380691648261`
- Selected schedule: `Selective`
- Mode reason: `selective_activation`
- Active ratio target: `0.10`
- Gamma range: `0.01` to `0.06`

Validation:

- Static MSE / MAE: `0.3589252558764385` / `0.24906166032113441`
- Post-hoc MSE / MAE: `0.35872791488831784` / `0.24869727953842521`
- Gain: MSE `+0.0549%`, MAE `+0.1463%`

Test:

- Static MSE / MAE: `0.3921327785795561` / `0.2687111022020055`
- Post-hoc MSE / MAE: `0.3921420741364507` / `0.2683736946179571`
- Gain: MSE `-0.0024%`, MAE `+0.1255%`

## Risk Localization Evidence

For the selected global validation active ratio `p=0.10`:

- Active windows: `167`
- Active fold coverage: `0.50`
- Active fold entropy norm: `0.13871325435016238`
- Effective active folds: `1.2120309241758773`
- Active fold concentration: `0.9520958083832335`

The active windows are therefore strongly concentrated rather than uniformly distributed. This supports a Traffic narrative of localized non-stationary risk detection, not broad validation-wide improvement.

The rolling-local diagnostic provides an ablation against this concentration:

- `global_val`, `p=0.10`: coverage `0.50`, entropy norm `0.1387`, effective folds `1.21`
- `centered_rolling_raw`, `W=168`, `p=0.10`: coverage `1.00`, entropy norm `0.9157`, effective folds `3.56`
- `centered_rolling_raw`, `W=336`, `p=0.10`: coverage `1.00`, entropy norm `0.9766`, effective folds `3.87`
- `centered_rolling_raw`, `W=504`, `p=0.10`: coverage `1.00`, entropy norm `0.8181`, effective folds `3.11`

Interpretation: shorter local calibration can reveal hidden local anomalies across all folds, but it weakens the strict global validation selection story. Keep this as a diagnostic ablation, not the main closed-loop selection rule.

## Graph-Structure Evolution Evidence

This analysis uses windowed ridge `DeltaA` on a fixed PCMCI support. It is not fold-specific PCMCI rediscovery.

Cross-variable graph summary:

- `fold1_all`: edge_count_mean `53.5313`, L1 mean `44.7907`, effective_targets `27.5590`
- `fold4_all`: edge_count_mean `136.8578`, L1 mean `137.9283`, effective_targets `5.3740`
- `fold4_active`: edge_count_mean `145.0440`, L1 mean `146.0973`, effective_targets `6.4065`
- `fold4_inactive`: edge_count_mean `131.7734`, L1 mean `132.8547`, effective_targets `4.3994`

Top-edge Jaccard:

- `fold1_all` vs `fold4_active`, top100: `0.0152`
- `fold1_all` vs `fold4_active`, top379: `0.0513`
- `fold4_inactive` vs `fold4_active`, top100: `0.3699`
- `fold4_inactive` vs `fold4_active`, top379: `0.4747`

Target focalization:

- In `fold4_active`, target `840` has mass share `0.5388453981167075`
- Target `840` active source count: `312`
- Top source for target `840`: `858`

Interpretation: Fold4 is not merely higher residual. It shows stronger cross-variable dynamic correction magnitude and target-level focalization on the fixed causal support.

## Files

Tables are under `tables/`.

Figures are under `figures/`.

Raw provenance references are under `raw_refs/`.

See `manifest.json` for source paths, file roles, and the intentionally excluded large array artifacts.

## Large Artifacts Excluded

The package intentionally excludes large `.npy` files from:

`C:\Users\cyl\Desktop\data\interfaces\Traffic_graph_interface_parcorr`

These arrays are required for recomputation but not for paper-facing mechanism evidence review. Their paths are recorded in `manifest.json`.
