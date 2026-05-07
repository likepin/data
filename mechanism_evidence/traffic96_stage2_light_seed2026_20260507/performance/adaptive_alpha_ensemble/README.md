# Traffic96 Stage2-Light Adaptive-Alpha Evidence

Generated: 2026-05-07

This package freezes the light Stage2 Traffic performance branch. Stage2-Light adds one paired
seed (`projection_3`, `seed=2026`) to the existing three baseline/staticcausal projections and
reruns validation-selected adaptive-alpha ensembling over 8 candidates.

## Boundary

This is a Traffic prediction-level performance evidence package. It is not a post-hoc dynamic CACI
closed-loop result and should not be used as evidence that dynamic `deltaA` calibration improved
Traffic forecasting.

## Selection

- Selected ensemble: `blend_baseline_static_alpha_variable_shrink`
- Selection reason: `best_val_mse_with_mae_guard`
- Reference best single: `static_p1`
- Candidate count: `8` (`baseline_p0..p3`, `static_p0..p3`)
- Test split used only once for final selected evaluation.

## Key Results

- Global closed-form alpha: `0.755954`
- Per-variable alpha mean/std: `0.784892 / 0.144948`
- Validation MSE/MAE: `0.349884 / 0.239219`
- Test MSE/MAE: `0.382640 / 0.259420`
- Test gain vs `static_p1`: MSE `+2.4370%`, MAE `+3.3201%`
- Increment vs Stage1.5 selected: MSE `+0.0782%`, MAE `+0.0949%`

## Negative Control

The shuffled-alpha control permutes the same 862 alpha values across targets. It preserves the
alpha distribution but breaks target identity.

- Shuffled median test MSE: `0.384140`
- Observed test MSE: `0.382640`
- Observed gain vs shuffled median: `+0.3905%`
- Lower-is-better test rank fraction among shuffles: `0.0000`

## Files

- `raw_outputs/`: direct small outputs from `traffic_existing_prediction_ensemble.py --tag stage2_light_seed2026`.
- `training_logs/`: Stage2 train/backfill commands and logs; no `.npy` arrays.
- `tables/traffic96_static_stage2_light_seed2026_frozen_table.csv`: frozen Stage2-Light comparison table.
- `tables/traffic96_static_stage2_light_seed2026_target_diagnostics.csv`: per-target alpha, gains, and PCMCI graph diagnostics.
- `tables/traffic96_static_stage2_light_seed2026_top_alpha_targets.csv`: highest-alpha targets for mechanism inspection.
- `tables/traffic96_static_stage2_light_seed2026_alignment_summary.csv`: correlation and negative-control summary rows.
- `tables/traffic96_static_stage2_light_seed2026_shuffled_negative_control.csv`: shuffled-alpha MSE diagnostics.
- `figures/traffic96_static_stage2_light_seed2026_alpha_distribution.png`: alpha distribution.
- `figures/traffic96_static_stage2_light_seed2026_alpha_gain_scatter.png`: alpha vs validation gain.
- `figures/traffic96_static_stage2_light_seed2026_alpha_graph_scatter.png`: alpha vs PCMCI parent strength.
