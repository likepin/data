# Traffic96 Mechanism and Performance Evidence Package

Generated: 2026-05-06

Data repo branch: `caci-posthoc-quality-guard`

Base commit: `3b366ea5c71c1a28cf0ee32f3fd0db76885b637c`

This package combines Traffic-96 mechanism evidence and Traffic-96 performance evidence. It is intentionally lightweight: only CSV, PNG, JSON, and Markdown files are copied. Large `.npy` arrays are referenced by provenance paths in `manifest.json`, but are not copied.

## Evidence Split

Use these results as three separate claims.

### 1. Mechanism Evidence

Location:

`mechanism/`

Claim:

The guarded CACI post-hoc closed loop can activate on high-dimensional Traffic after `log_tail_adaptive` lambda scaling, and the dynamic branch localizes a non-stationary risk episode with fixed-support graph-correction reconfiguration.

Key result:

- Selected lambda: `change_slope_no_range`, `window=40`, `k=2`
- Lambda scaling: `log_tail_adaptive`
- Lambda transform: `rank`
- Mode: `Selective`
- Mode reason: `selective_activation`
- Validation effect: `MSE +0.055%`, `MAE +0.146%`
- Test effect: `MSE -0.002%`, `MAE +0.126%`

Mechanism interpretation:

- Traffic is no longer rejected because of lambda saturation.
- Active windows are concentrated in a localized non-stationary period.
- Fold4 active windows show stronger cross-variable `DeltaA` correction and target-level focalization.
- Target `840` absorbs about `53.9%` of fold4-active cross-variable correction mass.

Do not claim:

This is not a strong Traffic forecasting-performance gain.

### 2. Post-Hoc Test-Oracle Upper Bound

Locations:

- `performance/test_oracle_val_energy/`
- `performance/test_oracle_test_energy/`

Claim:

The existing post-hoc Traffic dynamic asset has only a very small test-side upper bound, even under oracle selection.

Best test-oracle results:

- Val-energy target mask, best MSE: `MSE +0.0151%`, `MAE -0.0028%`
- Val-energy target mask, best MSE with `MAE >= 0`: `MSE +0.0132%`, `MAE +0.0003%`
- Test-energy target mask, best MSE: `MSE +0.0172%`, `MAE -0.0017%`
- Test-energy target mask, best MSE with `MAE >= 0`: `MSE +0.0149%`, `MAE +0.0013%`

Interpretation:

Post-hoc target/gamma tuning is not the right path for a strong Traffic performance result. The oracle result is diagnostic only and must not be reported as a valid test-selected model.

### 3. Existing-Prediction Ensemble Performance Evidence

Location:

`performance/ensemble/`

Claim:

Existing Traffic baseline and static-causal predictors have complementary errors. A validation-selected prediction-level ensemble provides the strongest Traffic performance result so far.

Selected ensemble:

- `blend_baseline_static_alpha_0.60`
- `40%` baseline mean
- `60%` static-causal mean
- baseline projection weights: `0.1333` each
- static-causal projection weights: `0.2000` each

Reference:

- Best validation single model: `static_p1`
- Reference validation `MSE / MAE = 0.357625 / 0.248371`
- Reference test `MSE / MAE = 0.392198 / 0.268328`

Selected ensemble:

- Validation `MSE / MAE = 0.351478 / 0.239588`
- Validation gain vs best single: `MSE +1.7188%`, `MAE +3.5363%`
- Test `MSE / MAE = 0.383882 / 0.259708`
- Test gain vs best single: `MSE +2.1203%`, `MAE +3.2126%`

Interpretation:

This is a Traffic performance branch, not evidence that post-hoc dynamic CACI calibration improves Traffic. Keep it separate from the main guarded post-hoc protocol.

## Recommended Paper Use

Main protocol table:

- Use the guarded post-hoc closed-loop result.
- Traffic status: `Selective / weak-or-neutral dynamic effect`.
- Use mechanism diagnostics to support non-stationary risk localization.

Traffic performance table or supplement:

- Use the existing-prediction ensemble result if a stronger Traffic number is needed.
- Report it as a simple validation-selected ensemble over existing baseline/static predictors.

Do not mix:

- Do not present the ensemble gain as post-hoc CACI dynamic calibration gain.
- Do not present test-oracle upper-bound results as formal model selection.

## Package Layout

`mechanism/tables/`

Mechanism CSVs from the guarded post-hoc closed loop and graph/risk diagnostics.

`mechanism/figures/`

Residual-complexity and active-window overlay plots.

`mechanism/raw_refs/`

Traffic interface manifest and the previous mechanism package manifest.

`performance/ensemble/`

Existing-prediction ensemble manifest, validation grid, selected weights, and selected test summary.

`performance/test_oracle_val_energy/`

Test-oracle diagnostic using validation-derived target energy masks.

`performance/test_oracle_test_energy/`

Pure test-oracle diagnostic using test-derived target energy masks.

## Large Artifacts Excluded

The package excludes all large `.npy` arrays, including:

- `C:\Users\cyl\Desktop\data\interfaces\Traffic_graph_interface_parcorr\deltaA_train.npy`
- `C:\Users\cyl\Desktop\data\interfaces\Traffic_graph_interface_parcorr\deltaA_val.npy`
- `C:\Users\cyl\Desktop\data\interfaces\Traffic_graph_interface_parcorr\deltaA_test.npy`
- Traffic prediction arrays under `C:\Users\cyl\Desktop\iTransformer-phasec-clean\results`

These files are required for recomputation but not for curated evidence review.
