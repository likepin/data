# Traffic96 Stage3.5 Negative Result Explanation

## Executive Summary

Traffic Stage3.5 is a weak positive result, but a negative mechanism result for the stronger high-risk-window claim.
The closed-form eta2 add-on slightly improves the Stage2 adaptive static anchor, yet Risk Windows show that the test gain is not produced by active high-gamma windows.

Core conclusion:

> Stage3.5 should be framed as a small guarded dynamic-aware add-on, not as evidence that lambda-gated dynamics reliably attack high-risk Traffic windows.

## Key Evidence

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

## Claim Verdict Matrix

| candidate_claim | evidence | verdict | paper_safe_framing |
| --- | --- | --- | --- |
| Stage3.5 is a strong Traffic performance module. | Closed-form eta2 test gain vs Stage2 is MSE +0.0704%, MAE +0.0961%. | reject_strong_claim | Stage3.5 provides a weak positive add-on over the adaptive static anchor. |
| Lambda-gated dynamics successfully localize high-risk windows on Traffic test. | test gamma_active_gt_floor MSE gain is -0.0343%; top_rank_5pct_gamma MSE gain is -0.0591%. | rejected_on_test | Current lambda-aware correction does not yet provide reliable high-risk-window localization. |
| The overall gain is driven by active high-gamma windows. | gamma_floor covers 90.54% of test windows and contributes 103.94% of SSE gain. | rejected_by_contribution | Traffic Stage3.5 gain is mostly a weak global / gamma-floor correction effect. |
| Validation Fold 4 evidence is enough to claim test-time risk localization. | Validation Fold 4 MSE gain is +0.5723%, but high-gamma active windows are negative on test. | not_generalized | Fold 4 is useful as anomaly evidence, but it does not justify a broad dynamic localization claim. |
| The dynamic branch should become the mainline for Traffic. | Stage2 adaptive-alpha remains the stable performance anchor; Stage3.5 adds only a very small post-hoc increment. | reject_mainline_shift | Keep static anchor as the main result; dynamic branch remains guarded and subordinate. |
| The post-hoc guards can be relaxed after Stage3.5. | Eta is clipped (`eta_raw=3.670469`, `eta_mult=2.0`) and high-gamma test windows are unstable. | reject_guard_relaxation | Stage3.5 supports guard necessity rather than guard relaxation. |

## Paper-Safe Framing

Use:
- `Traffic confirms that CACI's guarded post-hoc dynamic branch can produce a small additional correction after a strong adaptive static anchor.`
- `Risk-window diagnostics reveal a boundary condition: the current lambda-gated correction is not reliable in active high-gamma windows on the Traffic test split.`
- `This supports the paper's selective / guarded protocol: dynamic information should be admitted only under validation-selected and guard-constrained conditions.`

Avoid:
- `Traffic proves the dynamic branch is broadly strong.`
- `Lambda reliably localizes high-risk windows on Traffic test.`
- `High-gamma windows are the main source of the Traffic gain.`
- `Guards can be relaxed after Stage3.5.`

## Source Tables

- Frozen performance table: `C:\Users\cyl\Desktop\data\mechanism_evidence\traffic96_stage3_lambda_three_source_20260507\performance\stage3_lambda_three_source\tables\traffic96_static_stage3_lambda_three_source_frozen_table.csv`
- Risk group table: `C:\Users\cyl\Desktop\data\mechanism_evidence\traffic96_stage3_lambda_three_source_20260507\mechanism\risk_windows\traffic96_stage3_eta2_risk_group_table.csv`
- Fold contribution table: `C:\Users\cyl\Desktop\data\mechanism_evidence\traffic96_stage3_lambda_three_source_20260507\mechanism\risk_windows\traffic96_stage3_eta2_fold_contribution.csv`
- Risk Windows README: `C:\Users\cyl\Desktop\data\mechanism_evidence\traffic96_stage3_lambda_three_source_20260507\mechanism\risk_windows\README.md`
