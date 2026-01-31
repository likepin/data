# Selection Report

## Top-5 by score_equal

| rank | window | k | score_equal | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 3 | 4.739464 | 0.625838 | 0.070608 | -0.005197 | 0.003378 | 0.012853 | 0.029408 |
| 2 | 50 | 3 | 4.736876 | 0.600553 | 0.098619 | 0.019724 | 0.063866 | 0.020911 | 0.042639 |
| 3 | 50 | 2 | 4.100294 | 0.530795 | 0.014475 | -0.065916 | 0.006723 | 0.015006 | 0.030116 |
| 4 | 30 | 2 | 3.969873 | 0.499417 | 0.043808 | -0.076675 | 0.051926 | 0.024807 | 0.043753 |

## Top-5 by score_gating

| rank | window | k | score_gating | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 50 | 3 | 0.899789 | 0.600553 | 0.098619 | 0.019724 | 0.063866 | 0.020911 | 0.042639 |
| 2 | 80 | 3 | 0.855675 | 0.625838 | 0.070608 | -0.005197 | 0.003378 | 0.012853 | 0.029408 |
| 3 | 30 | 2 | 0.726349 | 0.499417 | 0.043808 | -0.076675 | 0.051926 | 0.024807 | 0.043753 |
| 4 | 50 | 2 | 0.657511 | 0.530795 | 0.014475 | -0.065916 | 0.006723 | 0.015006 | 0.030116 |

## Top-5 by score_regime

| rank | window | k | score_regime | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 50 | 3 | 0.899789 | 0.600553 | 0.098619 | 0.019724 | 0.063866 | 0.020911 | 0.042639 |
| 2 | 80 | 3 | 0.877584 | 0.625838 | 0.070608 | -0.005197 | 0.003378 | 0.012853 | 0.029408 |
| 3 | 50 | 2 | 0.727038 | 0.530795 | 0.014475 | -0.065916 | 0.006723 | 0.015006 | 0.030116 |
| 4 | 30 | 2 | 0.720148 | 0.499417 | 0.043808 | -0.076675 | 0.051926 | 0.024807 | 0.043753 |

## Top-5 consistency

- Common configs across all three top-5: [(30, 2), (50, 2), (50, 3), (80, 3)]

## Component Contributions

- contrib_equal.png
- contrib_gating.png
- contrib_regime.png

## Pareto Plots

- pareto_auc_vs_smooth.png
- pareto_corr_vs_smooth.png

## Conclusion Template

We first filtered unstable configurations by smoothness and correlation constraints, then ranked candidates by gating-friendly score. The top configuration balances regime separation (AUC/sep) and prediction consistency (corr_mse) while keeping lambda smoothness within acceptable bounds.
