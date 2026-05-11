| dataset | horizon | route_family | reference | mode | active_ratio | test_mse | test_mae | mse_gain_pct | mae_gain_pct | selection_reason | paper_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 96 | guarded_posthoc_dynamic | static_anchor | Selective | 0.081508 | 0.388088 | 0.405779 | 0.032453 | 0.064755 | one_se_and_double_guard | positive_but_small |
| Weather | 96 | guarded_posthoc_dynamic | static_anchor_patience3 | Selective | 0.050651 | 0.172459 | 0.212004 | 0.061977 | 0.008193 | one_se_and_double_guard | tiny_positive_guarded_dynamic |
| ECL | 96 | guarded_posthoc_dynamic | static_anchor | Bypass | 0.000000 | 0.144953 | 0.237570 | 0.000000 | 0.000000 | fallback_static_only | bypass_or_neutral |
| Solar-96 | 96 | guarded_posthoc_dynamic | static_anchor | Selective | 0.041183 | 0.204988 | 0.230618 | 0.252995 | 0.319040 | one_se_and_double_guard | positive_but_small |
| Solar-192 | 192 | guarded_posthoc_dynamic | static_anchor | Selective | 0.048154 | 0.243756 | 0.261294 | 0.057363 | 0.001253 | one_se_and_double_guard | positive_but_small |
| Traffic | 96 | guarded_posthoc_dynamic | static_anchor | Selective | 0.094638 | 0.392142 | 0.268374 | -0.002376 | 0.125549 | one_se_and_double_guard | mixed_metric |
| ETTh1 | 96 | stage3_lambda_three_source | adaptive_anchor | stage3_closed_form_all |  | 0.381877 | 0.399841 | -0.425027 | -0.085516 | best_val_mse_with_mae_guard | negative_addon |
| Solar-96 | 96 | stage3_lambda_three_source | adaptive_anchor | stage3_closed_form_top_alpha_5 |  | 0.195922 | 0.226946 | 0.045074 | 0.024966 | best_val_mse_with_mae_guard | weak_positive_addon |
| Solar-192 | 192 | stage3_lambda_three_source | adaptive_anchor | stage2_anchor |  | 0.233232 | 0.254287 | 0.000000 | 0.000000 | best_val_mse_with_mae_guard | fallback_to_anchor |
| Traffic | 96 | stage3_lambda_three_source | adaptive_anchor | Stage3 lambda three-source, closed-form eta2 |  | 0.382371 | 0.259170 | 0.070371 | 0.096147 |  | weak_positive_addon |
| Weather | 96 | mse_primary_target_gate | audit_adaptive_anchor_not_pat3_headline | target_gate_g20_d40 | 0.400000 | 0.169689 | 0.210608 | 0.066402 | -0.027601 | best_val_mse_relaxed_mae_or_fold_guard | audit_only_mse_positive_mae_negative |
| Solar-96 | 96 | mse_primary_target_gate | adaptive_anchor | target_gate_g10_d20 | 0.200000 | 0.195569 | 0.226831 | 0.225171 | 0.075641 | best_val_mse_relaxed_mae_or_fold_guard | mse_primary_positive_audit |
| Solar-192 | 192 | mse_primary_target_gate | adaptive_anchor | target_gate_g10_d20 | 0.200000 | 0.233103 | 0.254251 | 0.055094 | 0.014087 | best_val_mse_relaxed_mae_or_fold_guard | mse_primary_positive_audit |
