# Weather-96 MSE-Primary Target-Gated Dynamic Route

| route | variant | selected | test MSE | test MAE | test MSE vs adaptive | test MAE vs adaptive | selection |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| adaptive_anchor | per_variable_shrinkage_alpha | stage2_anchor | 0.169801 | 0.210550 | +0.0000% | +0.0000% | stage2_anchor_reference |
| strict_target_gate | static_p0 | stage2_anchor | 0.169801 | 0.210550 | +0.0000% | +0.0000% | fallback_stage2_anchor_no_fold_stable_candidate |
| strict_target_gate | static_mean | stage2_anchor | 0.169801 | 0.210550 | +0.0000% | +0.0000% | fallback_stage2_anchor_no_fold_stable_candidate |
| mse_primary_target_gate | static_p0 | target_gate_g20_d40 | 0.169696 | 0.210596 | +0.0621% | -0.0218% | best_val_mse_relaxed_mae_or_fold_guard |
| mse_primary_target_gate | static_mean | target_gate_g20_d40 | 0.169689 | 0.210608 | +0.0664% | -0.0276% | best_val_mse_relaxed_mae_or_fold_guard |
