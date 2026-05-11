# Weather96 Dynamic Gate Negative Evidence

Generated: 2026-05-11

This package freezes the Weather-96 diagnostic evidence for why the current lambda/dynamic branch should not be promoted to a standalone performance route.
It complements the Weather MSE-primary target-gate performance package by isolating the dynamic-gate failure mode.

## Boundary

- This is a mechanism and risk diagnostic package, not a new forecasting result.
- Validation is used for fitting the lightweight probes; test rows are diagnostic readouts.
- The large `deltaA_signal_audit` source directories remain local artifacts and are referenced in `manifest.json`.

## Executive Summary

The Weather-96 dynamic branch contains weak recoverable signal under ideal scaling, but the uncalibrated dynamic increment is mostly harmful.
Logistic and gain-aware probes can rank risk and reduce expected damage, yet Ridge top-k selections remain negative on average and Huber mostly chooses zero-dynamic bypass regions.

Core interpretation:

> Weather-96 supports the CACI guard philosophy: dynamic information is detectable, but forced dynamic activation is unsafe; static/adaptive anchor should remain primary.

## Evidence Snapshot

| evidence_id | split | metric | value | interpretation |
| --- | --- | --- | --- | --- |
| lambda_alignment_test | test | lambda/gamma vs unit oracle MSE gain | lambda Spearman -0.253034; gamma Spearman 0.009317 | Current lambda is weakly aligned with where raw dynamic correction is truly beneficial. |
| unit_dynamic_gain_test | test | raw unit dynamic gain | mean -0.228528; positive-rate 3.782% | The uncalibrated dynamic increment is mostly harmful on Weather-96. |
| oracle_eta2_gain_test | test | oracle eta2 gain | mean 0.005348 | There is weak recoverable signal under ideal scaling, but the magnitude is small. |
| ridge_gain_regression_test | test | gain regression rank quality | Pearson 0.855251; Spearman 0.473427; R2 0.715685 | Ridge learns a strong continuous gain ranking, so the probe is informative. |
| huber_gain_regression_test | test | robust gain regression rank quality | Pearson 0.860943; Spearman 0.741797; R2 0.740570 | Huber learns a conservative risk-avoidance ordering. |
| ridge_top10_test | test | Ridge top-10% risk-return | mean -0.032225; positive-rate 24.886%; worst5 -0.390679 | Ridge reduces expected loss sharply but does not turn the selected set positive. |
| ridge_top1_test | test | Ridge top-1% risk-return | mean -0.048829; positive-rate 21.113%; worst5 -0.457355 | Even the most optimistic Ridge slice remains negative on average. |
| huber_top10_test | test | Huber top-10% risk-return | mean 0.000001; positive-rate 0.078%; nonzero dynamics approximately zero in top bin | Huber mainly selects zero-dynamic / zero-gain windows, i.e. safe bypass behavior. |
| ridge_bin_contrast | test | Ridge top vs bottom decile | top decile mean -0.032223; bottom decile mean -1.403961 | The ranking separates catastrophic negative windows from less harmful windows. |
| huber_zero_region | test | Huber top decile | mean 0.000001; nonzero-dynamic-rate 0.155% | The robust route identifies a near-bypass safety zone rather than an active gain zone. |
| val_test_consistency | val/test | raw dynamic positive-rate | val 6.493%; test 3.782% | The positive raw-dynamic region is sparse on both splits. |
| top_coefficients | val-fit | dominant gain-regression features | huber:dynamic_energy_target=-1.239613; huber:dynamic_abs_mean_target=-0.009319; ridge:dynamic_energy_target=-1.328644; ridge:dynamic_abs_mean_target=0.158931 | Dynamic energy/shape dominate gain prediction; lambda_rank is not the main signal. |

## Claim Verdict Matrix

| candidate_claim | evidence | verdict | paper_safe_framing |
| --- | --- | --- | --- |
| Weather-96 dynamic branch can be made a stable positive performance route by better lambda gating. | mean -0.032225; positive-rate 24.886%; worst5 -0.390679 | reject_for_now | Dynamic correction is diagnosable but should remain guard-suppressed on Weather-96. |
| The current lambda_rank is the right primary gate signal. | lambda Spearman -0.253034; gamma Spearman 0.009317 | reject | lambda_rank is a weak risk proxy; dynamic energy/shape carries more diagnostic information. |
| A probability gate is sufficient. | Logistic probe improves positive-rate, but gain-aware top-k remains negative on average. | reject | Expected gain and downside risk must be audited, not only hit probability. |
| Gain-aware regression proves a deployable positive dynamic route. | Pearson 0.855251; Spearman 0.473427; R2 0.715685 / mean -0.048829; positive-rate 21.113%; worst5 -0.457355 | not_supported | Gain regression is useful for risk ordering, but not sufficient for positive Weather-96 deployment. |
| Huber can rescue the dynamic branch. | mean 0.000001; positive-rate 0.078%; nonzero dynamics approximately zero in top bin | reject_as_gain_route | Huber behaves as a conservative bypass selector. |
| Weather-96 should be used as a negative mechanism case. | mean -0.228528; positive-rate 3.782% / mean 0.005348 | support | Weather-96 supports the guard philosophy: dynamic information exists, but forced activation is unsafe. |

## Key Files

- `tables/weather96_dynamic_gate_evidence_snapshot.csv`: compact evidence rows.
- `tables/weather96_dynamic_gate_claim_verdict_matrix.csv`: claim-safe interpretation matrix.
- `tables/weather96_static_pat3_lambda_adequacy_split_summary.csv`: lambda adequacy split summary.
- `tables/weather96_static_pat3_lambda_gate_logistic_probe_target_gain_metrics.csv`: gain regression metrics.
- `tables/weather96_static_pat3_lambda_gate_logistic_probe_target_gain_topk_cvar.csv`: top-k and CVaR table.
- `tables/weather96_static_pat3_lambda_gate_logistic_probe_target_gain_quantile_bins.csv`: predicted-gain decile bins.
- `figures/*risk_return_frontier.png`: risk-return frontier plots.
- `figures/*top5_gain_distribution.png`: selected top-5% gain distribution diagnostics.

## Source Pointers

- adequacy source: `C:\Users\cyl\Desktop\data\deltaA_signal_audit\weather96_static_pat3_lambda_adequacy`
- probe source: `C:\Users\cyl\Desktop\data\deltaA_signal_audit\weather96_static_pat3_lambda_gate_logistic_probe`
- git head: `858bb65c5158eed2921d10b738fe8fdd7d7b428c`
