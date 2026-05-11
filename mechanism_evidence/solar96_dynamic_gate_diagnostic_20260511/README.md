# Solar96 Dynamic Gate Diagnostic Evidence

Generated: 2026-05-11

This package freezes the Solar-96 dynamic-gate diagnostic readout using the same lambda adequacy and gain-aware/CVaR probe protocol used for Weather-96.

## Boundary

- This is a mechanism and risk diagnostic package, not a new training result.
- Validation is used for fitting lightweight probes; test rows are diagnostic readouts.
- The source `deltaA_signal_audit` directories remain local artifacts and are referenced in `manifest.json`.

## Executive Summary

Solar-96 has clearer recoverable dynamic signal than Weather-96 under oracle scaling, but the current deployable lambda/gain gates still do not produce a positive active frontier.
Ridge and Huber can rank safer windows, yet their top selections are zero-gain or negative on average; this supports guarded selective/bypass behavior rather than forced activation.

Core interpretation:

> Solar-96 is a medium-scale mixed case: dynamic information is stronger than Weather, but current gate design is still not strong enough for an active performance claim.

## Evidence Snapshot

| evidence_id | split | metric | value | interpretation |
| --- | --- | --- | --- | --- |
| lambda_alignment_test | test | lambda/gamma vs unit oracle MSE gain | lambda Spearman 0.131329; gamma Spearman 0.097925 | Solar lambda is weakly positive on test, but not strong enough to select a deployable active route. |
| unit_dynamic_gain_test | test | raw unit dynamic gain | mean -0.310923; positive-rate 8.726% | The uncalibrated dynamic increment is harmful on average despite a slightly larger positive region than Weather. |
| oracle_eta2_gain_test | test | oracle eta2 gain | mean 0.032410 | Solar has clearer recoverable dynamic signal under ideal scaling than Weather. |
| selected_gamma_test | test | selected gamma gain | mean 0.000522; active-ratio 4.118% | The existing closed-loop schedule is safe but very weak on test. |
| ridge_gain_regression_test | test | gain regression generalization | Pearson 0.124157; Spearman 0.478146; R2 -0.187258 | Ridge preserves some ranking but its continuous gain calibration does not generalize cleanly. |
| huber_gain_regression_test | test | robust gain regression generalization | Pearson 0.035154; Spearman 0.479935; R2 -0.503421 | Huber behaves as a conservative ranker but not a positive gain estimator. |
| ridge_top5_test | test | Ridge top-5% risk-return | mean 0.000000; positive-rate 0.000%; worst5 0.000000 | Top-5% Ridge mostly selects zero-gain/bypass-like rows, not positive dynamic gain. |
| ridge_top10_test | test | Ridge top-10% risk-return | mean -0.073537; positive-rate 0.704%; worst5 -0.359927 | Ridge top-10% remains negative, so the active gain frontier is not deployable. |
| huber_top10_test | test | Huber top-10% risk-return | mean -0.076210; positive-rate 0.438%; worst5 -0.314692 | Huber avoids the worst loss only by staying near bypass/zero-dynamic behavior. |
| ridge_bin_contrast | test | Ridge top vs bottom decile | top decile mean -0.073537; bottom decile mean -0.434932 | The risk ranker separates safer windows from catastrophic windows, but top decile is still negative. |
| huber_zero_region | test | Huber top decile | mean -0.076210; nonzero-dynamic-rate 35.846% | The robust route selects a mixed zero/safe region rather than a reliable active gain region. |
| val_test_consistency | val/test | raw dynamic positive-rate | val 6.917%; test 8.726% | The positive raw-dynamic region is sparse but slightly denser than Weather. |
| top_coefficients | val-fit | dominant gain-regression features | huber:dynamic_energy_target=-0.198975; huber:lambda_rank=-0.084546; ridge:dynamic_abs_mean_target=-0.129545; ridge:dynamic_energy_target=-0.067194 | Dynamic energy/shape dominate gain prediction; static alpha is not the main driver. |

## Claim Verdict Matrix

| candidate_claim | evidence | verdict | paper_safe_framing |
| --- | --- | --- | --- |
| Solar-96 has stronger dynamic signal than Weather-96. | mean 0.032410 / val 6.917%; test 8.726% | support_with_guard | Solar shows clearer recoverable dynamic signal, but only under ideal scaling or heavy guard. |
| Solar-96 dynamic branch can be directly promoted to a positive active route. | mean -0.073537; positive-rate 0.704%; worst5 -0.359927 | reject_for_now | Current deployable gain-aware gates do not produce a positive active frontier. |
| The existing closed-loop schedule is useful but weak. | mean 0.000522; active-ratio 4.118% | support | Closed-loop scheduling contributes a tiny safe correction rather than a strong dynamic route. |
| Risk-return diagnostics justify bypass/guard behavior on Solar. | top decile mean -0.073537; bottom decile mean -0.434932 / mean -0.076210; nonzero-dynamic-rate 35.846% | support | Gain-aware probes identify safer windows but not reliable positive active corrections. |
| A probability gate alone is enough. | Target logistic AUC is useful, but top-k gain/CVaR remains non-positive. | reject | Solar reinforces the need for expected-gain and downside-risk audits. |
| Solar is a better next target than Traffic for refining dynamic gates. | mean 0.032410 and tractable 137-variable target-wise diagnostics. | support | Solar is the appropriate medium-scale case for dynamic-gate diagnostics before Traffic-scale deployment. |

## Key Files

- `tables/solar96_dynamic_gate_evidence_snapshot.csv`: compact evidence rows.
- `tables/solar96_dynamic_gate_claim_verdict_matrix.csv`: claim-safe interpretation matrix.
- `tables/solar96_static_lambda_adequacy_split_summary.csv`: lambda adequacy split summary.
- `tables/solar96_static_lambda_gate_logistic_probe_target_gain_metrics.csv`: gain regression metrics.
- `tables/solar96_static_lambda_gate_logistic_probe_target_gain_topk_cvar.csv`: top-k and CVaR table.
- `tables/solar96_static_lambda_gate_logistic_probe_target_gain_quantile_bins.csv`: predicted-gain decile bins.
- `figures/*risk_return_frontier.png`: risk-return frontier plots.
- `figures/*top5_gain_distribution.png`: selected top-5% gain distribution diagnostics.

## Source Pointers

- adequacy source: `C:\Users\cyl\Desktop\data\deltaA_signal_audit\solar96_static_lambda_adequacy`
- probe source: `C:\Users\cyl\Desktop\data\deltaA_signal_audit\solar96_static_lambda_gate_logistic_probe`
- git head: `05071950c591215c45e22c82b18a6f8b1908e5c9`
