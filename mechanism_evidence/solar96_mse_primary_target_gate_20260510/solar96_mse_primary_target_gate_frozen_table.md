# Solar-96 MSE-Primary Target-Gated Dynamic Route

| route | variant | selected | test MSE | test MAE | test MSE vs adaptive | test MAE vs adaptive | val MSE vs adaptive | val MAE vs adaptive | selection |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| adaptive_anchor | per_variable_shrinkage_alpha | blend_baseline_static_alpha_variable_shrink | 0.196010 | 0.227003 | +0.0000% | +0.0000% | +0.0000% | +0.0000% | best_val_mse_with_mae_guard |
| strict_target_gate | static_p0 | stage2_anchor | 0.196010 | 0.227003 | +0.0000% | +0.0000% | +0.0000% | +0.0000% | fallback_stage2_anchor_no_fold_stable_candidate |
| strict_target_gate | static_mean | stage2_anchor | 0.196010 | 0.227003 | +0.0000% | +0.0000% | +0.0000% | +0.0000% | fallback_stage2_anchor_no_fold_stable_candidate |
| mse_primary_target_gate | static_p0 | target_gate_g10_d20 | 0.195569 | 0.226831 | +0.2252% | +0.0756% | +0.0555% | -0.0372% | best_val_mse_relaxed_mae_or_fold_guard |
| mse_primary_target_gate | static_mean | target_gate_g10_d20 | 0.195623 | 0.226852 | +0.1974% | +0.0667% | +0.0401% | -0.0407% | best_val_mse_relaxed_mae_or_fold_guard |
