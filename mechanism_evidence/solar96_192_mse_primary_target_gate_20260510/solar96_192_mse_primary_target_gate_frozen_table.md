# Solar-96/192 MSE-Primary Target-Gated Dynamic Route

| dataset | horizon | route | variant | selected | test MSE | test MAE | test MSE vs adaptive | test MAE vs adaptive | selection |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Solar-96 | 96 | adaptive_anchor | per_variable_shrinkage_alpha | blend_baseline_static_alpha_variable_shrink | 0.196010 | 0.227003 | +0.0000% | +0.0000% | best_val_mse_with_mae_guard |
| Solar-96 | 96 | strict_target_gate | static_p0 | stage2_anchor | 0.196010 | 0.227003 | +0.0000% | +0.0000% | fallback_stage2_anchor_no_fold_stable_candidate |
| Solar-96 | 96 | strict_target_gate | static_mean | stage2_anchor | 0.196010 | 0.227003 | +0.0000% | +0.0000% | fallback_stage2_anchor_no_fold_stable_candidate |
| Solar-96 | 96 | mse_primary_target_gate | static_p0 | target_gate_g10_d20 | 0.195569 | 0.226831 | +0.2252% | +0.0756% | best_val_mse_relaxed_mae_or_fold_guard |
| Solar-96 | 96 | mse_primary_target_gate | static_mean | target_gate_g10_d20 | 0.195623 | 0.226852 | +0.1974% | +0.0667% | best_val_mse_relaxed_mae_or_fold_guard |
| Solar-192 | 192 | adaptive_anchor | per_variable_shrinkage_alpha | blend_baseline_static_alpha_variable_shrink | 0.233232 | 0.254287 | +0.0000% | +0.0000% | best_val_mse_with_mae_guard |
| Solar-192 | 192 | strict_target_gate | static_p0 | stage2_anchor | 0.233232 | 0.254287 | +0.0000% | +0.0000% | fallback_stage2_anchor_no_fold_stable_candidate |
| Solar-192 | 192 | strict_target_gate | static_mean | stage2_anchor | 0.233232 | 0.254287 | +0.0000% | +0.0000% | fallback_stage2_anchor_no_fold_stable_candidate |
| Solar-192 | 192 | mse_primary_target_gate | static_p0 | target_gate_g10_d20 | 0.233143 | 0.254264 | +0.0380% | +0.0091% | best_val_mse_relaxed_mae_or_fold_guard |
| Solar-192 | 192 | mse_primary_target_gate | static_mean | target_gate_g10_d20 | 0.233103 | 0.254251 | +0.0551% | +0.0141% | best_val_mse_relaxed_mae_or_fold_guard |
