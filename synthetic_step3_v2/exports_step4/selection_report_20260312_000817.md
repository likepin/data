# Selection Report

## Top-5 by score_equal

| rank | window | k | score_equal | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 120 | 2 | 7.308814 | 0.542572 | -0.002377 | -0.079350 | 0.025510 | 0.007312 | 0.016651 |
| 2 | 100 | 2 | 7.229635 | 0.545139 | -0.004340 | -0.076751 | 0.005085 | 0.008226 | 0.018001 |
| 3 | 80 | 2 | 7.223457 | 0.544022 | -0.000969 | -0.076844 | 0.000000 | 0.009661 | 0.021167 |
| 4 | 60 | 2 | 7.147230 | 0.536834 | 0.007496 | -0.067402 | 0.006734 | 0.012439 | 0.026065 |
| 5 | 50 | 2 | 7.063980 | 0.531360 | 0.014984 | -0.064766 | 0.006723 | 0.015011 | 0.030128 |

## Top-5 by score_gating

| rank | window | k | score_gating | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 4 | 0.797664 | 0.611764 | 0.097493 | 0.025447 | 0.040541 | 0.015305 | 0.034825 |
| 2 | 40 | 2 | 0.742468 | 0.521687 | 0.026375 | -0.068652 | 0.026846 | 0.019003 | 0.035183 |
| 3 | 50 | 2 | 0.730518 | 0.531360 | 0.014984 | -0.064766 | 0.006723 | 0.015011 | 0.030128 |
| 4 | 60 | 2 | 0.723605 | 0.536834 | 0.007496 | -0.067402 | 0.006734 | 0.012439 | 0.026065 |
| 5 | 120 | 2 | 0.715953 | 0.542572 | -0.002377 | -0.079350 | 0.025510 | 0.007312 | 0.016651 |

## Top-5 by score_regime

| rank | window | k | score_regime | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 2 | 0.882506 | 0.544022 | -0.000969 | -0.076844 | 0.000000 | 0.009661 | 0.021167 |
| 2 | 60 | 2 | 0.878659 | 0.536834 | 0.007496 | -0.067402 | 0.006734 | 0.012439 | 0.026065 |
| 3 | 50 | 2 | 0.876030 | 0.531360 | 0.014984 | -0.064766 | 0.006723 | 0.015011 | 0.030128 |
| 4 | 100 | 2 | 0.870391 | 0.545139 | -0.004340 | -0.076751 | 0.005085 | 0.008226 | 0.018001 |
| 5 | 120 | 2 | 0.870310 | 0.542572 | -0.002377 | -0.079350 | 0.025510 | 0.007312 | 0.016651 |

## Top-5 consistency

- Common configs across all three top-5: [(50, 2), (60, 2), (120, 2)]

## Component Contributions

- contrib_equal.png
- contrib_gating.png
- contrib_regime.png

## Pareto Plots

- pareto_auc_vs_smooth.png
- pareto_corr_vs_smooth.png

## Conclusion Template

We first filtered unstable configurations by smoothness and correlation constraints, then ranked candidates by gating-friendly score. The top configuration balances regime separation (AUC/sep) and prediction consistency (corr_mse) while keeping lambda smoothness within acceptable bounds.
