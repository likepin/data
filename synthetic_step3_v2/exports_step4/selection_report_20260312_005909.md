# Selection Report

## Top-5 by score_equal

| rank | window | k | score_equal | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 120 | 8 | 7.580384 | 0.688863 | 0.149610 | 0.136010 | 0.258503 | 0.016307 | 0.032612 |
| 2 | 120 | 6 | 7.563159 | 0.606715 | 0.061940 | 0.021310 | 0.061224 | 0.010898 | 0.020982 |
| 3 | 120 | 5 | 7.512995 | 0.578503 | 0.044363 | -0.015383 | 0.061224 | 0.011313 | 0.021528 |
| 4 | 60 | 2 | 6.764728 | 0.503435 | 0.010687 | -0.108112 | 0.035354 | 0.013551 | 0.024577 |
| 5 | 40 | 2 | 6.689681 | 0.481488 | 0.041813 | -0.095903 | 0.075503 | 0.022200 | 0.036871 |

## Top-5 by score_gating

| rank | window | k | score_gating | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 120 | 8 | 0.924639 | 0.688863 | 0.149610 | 0.136010 | 0.258503 | 0.016307 | 0.032612 |
| 2 | 120 | 6 | 0.827265 | 0.606715 | 0.061940 | 0.021310 | 0.061224 | 0.010898 | 0.020982 |
| 3 | 120 | 5 | 0.790982 | 0.578503 | 0.044363 | -0.015383 | 0.061224 | 0.011313 | 0.021528 |
| 4 | 80 | 4 | 0.775520 | 0.577273 | 0.068741 | -0.033195 | 0.021959 | 0.014571 | 0.027443 |
| 5 | 40 | 2 | 0.716927 | 0.481488 | 0.041813 | -0.095903 | 0.075503 | 0.022200 | 0.036871 |

## Top-5 by score_regime

| rank | window | k | score_regime | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 120 | 8 | 0.935255 | 0.688863 | 0.149610 | 0.136010 | 0.258503 | 0.016307 | 0.032612 |
| 2 | 120 | 6 | 0.913674 | 0.606715 | 0.061940 | 0.021310 | 0.061224 | 0.010898 | 0.020982 |
| 3 | 120 | 5 | 0.902654 | 0.578503 | 0.044363 | -0.015383 | 0.061224 | 0.011313 | 0.021528 |
| 4 | 80 | 4 | 0.854194 | 0.577273 | 0.068741 | -0.033195 | 0.021959 | 0.014571 | 0.027443 |
| 5 | 120 | 4 | 0.834266 | 0.540857 | 0.021765 | -0.054578 | 0.027211 | 0.009739 | 0.018785 |

## Top-5 consistency

- Common configs across all three top-5: [(120, 5), (120, 6), (120, 8)]

## Component Contributions

- contrib_equal.png
- contrib_gating.png
- contrib_regime.png

## Pareto Plots

- pareto_auc_vs_smooth.png
- pareto_corr_vs_smooth.png

## Conclusion Template

We first filtered unstable configurations by smoothness and correlation constraints, then ranked candidates by gating-friendly score. The top configuration balances regime separation (AUC/sep) and prediction consistency (corr_mse) while keeping lambda smoothness within acceptable bounds.
