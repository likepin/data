# Current Frozen CACI Tables 2026-05-11

This package freezes the current route-separated CACI result tables after the Weather/Solar dynamic-gate adequacy diagnostics.

## Scope

- Static and adaptive-fusion numbers are performance routes.
- Guarded post-hoc dynamic numbers are measured against the static anchor.
- Stage3 / target-gated dynamic numbers are measured against the adaptive anchor when available.
- Weather uses the paper-aligned `patience=3` summary as the canonical route table source.
- Weather MSE-primary target-gate is retained only as an audit row because its adaptive-anchor source is not the latest `patience=3` headline table.

## Main Readout

- The strongest current performance source is `adaptive_fusion`, not the dynamic graph.
- Guarded post-hoc dynamic gains are usually `0%` to `0.3%`.
- Stage3 lambda-aware add-ons are weak-positive on Traffic/Solar-96, fallback on Solar-192, and negative on ETTh1.
- The safe paper claim is route-separated: static/adaptive routes are performance routes; lambda/dynamic routes are guarded, optional, and dataset-sensitive.

## Files

- `current_route_table_full.csv/md`: numeric route-separated table.
- `current_route_table_paper.csv/md`: readable paper-facing route table.
- `dynamic_lambda_increment_full.csv/md`: dynamic/lambda increment and boundary table.
- `dynamic_lambda_increment_paper.csv/md`: readable dynamic/lambda boundary table.
- `manifest.json`: source file paths and git commit used for this freeze.

## Quick Dynamic Summary

Canonical guarded post-hoc MSE gains vs static anchor:

| dataset | horizon | mode | mse_gain_pct | mae_gain_pct | paper_status |
| --- | --- | --- | --- | --- | --- |
| ETTh1 | 96 | Selective | 0.032453 | 0.064755 | positive_but_small |
| Weather | 96 | Selective | 0.061977 | 0.008193 | tiny_positive_guarded_dynamic |
| ECL | 96 | Bypass | 0.000000 | 0.000000 | bypass_or_neutral |
| Solar-96 | 96 | Selective | 0.252995 | 0.319040 | positive_but_small |
| Solar-192 | 192 | Selective | 0.057363 | 0.001253 | positive_but_small |
| Traffic | 96 | Selective | -0.002376 | 0.125549 | mixed_metric |


Stage3 MSE gains vs adaptive anchor:

| dataset | horizon | mode | mse_gain_pct | mae_gain_pct | paper_status |
| --- | --- | --- | --- | --- | --- |
| ETTh1 | 96 | stage3_closed_form_all | -0.425027 | -0.085516 | negative_addon |
| Solar-96 | 96 | stage3_closed_form_top_alpha_5 | 0.045074 | 0.024966 | weak_positive_addon |
| Solar-192 | 192 | stage2_anchor | 0.000000 | 0.000000 | fallback_to_anchor |
| Traffic | 96 | Stage3 lambda three-source, closed-form eta2 | 0.070371 | 0.096147 | weak_positive_addon |

