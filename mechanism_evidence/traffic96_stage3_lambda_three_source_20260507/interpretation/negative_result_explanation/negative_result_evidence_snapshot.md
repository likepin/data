| evidence_id | split | metric | value | interpretation |
| --- | --- | --- | --- | --- |
| performance_stage2_anchor | test | Stage2 anchor MSE / MAE | 0.3826398812 / 0.2594195388 | Static/adaptive anchor is the stable Traffic performance base. |
| performance_stage3_grid | test | Stage3 grid gain vs Stage2 | MSE +0.0544%, MAE +0.0938% | Grid Stage3 is weak positive, not a strong new branch. |
| performance_stage35_eta2 | test | Stage3.5 closed-form eta2 gain vs Stage2 | MSE +0.0704%, MAE +0.0961% | Closed-form eta2 slightly improves grid, but the increment remains tiny. |
| risk_all_test | test | Overall risk-window gain | MSE +0.0704% | The whole Stage3.5 effect is weak positive. |
| risk_gamma_floor_test | test | gamma_floor coverage / SSE gain share | 90.54% / 103.94% | Most test gain comes from gamma-floor windows, not active high-risk windows. |
| risk_active_test | test | gamma_active_gt_floor MSE gain | -0.0343% | Active gamma windows are negative on test. |
| risk_top5_test | test | top_rank_5pct_gamma MSE gain | -0.0591% | The strongest high-gamma windows do not generalize as a positive mechanism. |
| risk_active_val | val | gamma_active_gt_floor MSE gain | +1.2558% | Validation suggests a local active-window opportunity. |
| risk_top5_val | val | top_rank_5pct_gamma MSE gain | +2.3662% | The validation-side high-gamma signal is real but not test-stable. |
| fold4_val | val | Validation Fold 4 MSE gain | +0.5723% | Fold 4 is an anomaly-sensitive validation region, not sufficient test evidence. |
