# Traffic96 Adaptive-Alpha Ensemble Evidence

Generated: 2026-05-06

This subpackage extends the existing Traffic prediction-level ensemble with validation-estimated adaptive alpha.

## Selection

- Selected ensemble: `blend_baseline_static_alpha_variable_shrink`
- Selection rule: validation MSE with non-negative MAE gain guard.
- Test split is used only once for final evaluation.

## Key Results

- Global closed-form alpha: `0.624119`
- Per-variable alpha mean/std: `0.668060 / 0.148481`
- Validation MSE/MAE: `0.350266 / 0.239480`
- Test MSE/MAE: `0.382939 / 0.259666`
- Test gain vs best single: MSE `+2.3606%`, MAE `+3.2283%`

## Negative Control

The shuffled-alpha negative control randomly permutes the same 862 alpha values across targets. It preserves the alpha distribution but breaks target identity.

- Shuffled median test MSE: `0.384043`
- Observed test MSE: `0.382939`
- Observed gain vs shuffled median: `+0.2875%`

## Files

- `raw_outputs/`: direct outputs from `traffic_existing_prediction_ensemble.py --tag adaptive_alpha`.
- `tables/traffic96_static_adaptive_alpha_target_diagnostics.csv`: per-target alpha, validation/test gains, and PCMCI graph metrics.
- `tables/traffic96_static_adaptive_alpha_top_alpha_targets.csv`: highest-alpha targets for mechanism inspection.
- `tables/traffic96_static_adaptive_alpha_alignment_summary.csv`: Spearman/Pearson alignment diagnostics.
- `tables/traffic96_static_adaptive_alpha_shuffled_negative_control.csv`: shuffled-alpha MSE diagnostics.
- `figures/traffic96_static_adaptive_alpha_alpha_distribution.png`: alpha distribution.
- `figures/traffic96_static_adaptive_alpha_alpha_gain_scatter.png`: alpha vs validation gain.
- `figures/traffic96_static_adaptive_alpha_alpha_graph_scatter.png`: alpha vs PCMCI parent strength.

## Reporting Boundary

Use this as Traffic performance evidence. Do not describe it as post-hoc dynamic CACI calibration gain.
