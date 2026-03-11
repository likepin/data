# Selection Report

## Top-5 by score_equal

| rank | window | k | score_equal | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 50 | 3 | 7.288179 | 0.600553 | 0.098619 | 0.019724 | 0.063866 | 0.020911 | 0.042639 |
| 2 | 80 | 2 | 7.222625 | 0.544022 | -0.000969 | -0.076844 | 0.000000 | 0.009661 | 0.021167 |
| 3 | 120 | 2 | 7.205489 | 0.542572 | -0.002377 | -0.079350 | 0.025510 | 0.007312 | 0.016651 |
| 4 | 50 | 2 | 7.068294 | 0.530795 | 0.014475 | -0.065916 | 0.006723 | 0.015006 | 0.030116 |
| 5 | 30 | 2 | 6.858040 | 0.499417 | 0.043808 | -0.076675 | 0.051926 | 0.024807 | 0.043753 |

## Top-5 by score_gating

| rank | window | k | score_gating | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 50 | 3 | 0.875919 | 0.600553 | 0.098619 | 0.019724 | 0.063866 | 0.020911 | 0.042639 |
| 2 | 30 | 2 | 0.772871 | 0.499417 | 0.043808 | -0.076675 | 0.051926 | 0.024807 | 0.043753 |
| 3 | 50 | 2 | 0.735217 | 0.530795 | 0.014475 | -0.065916 | 0.006723 | 0.015006 | 0.030116 |
| 4 | 80 | 3 | 0.730814 | 0.625838 | 0.070608 | -0.005197 | 0.003378 | 0.012853 | 0.029408 |
| 5 | 80 | 2 | 0.716802 | 0.544022 | -0.000969 | -0.076844 | 0.000000 | 0.009661 | 0.021167 |

## Top-5 by score_regime

| rank | window | k | score_regime | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 2 | 0.864886 | 0.544022 | -0.000969 | -0.076844 | 0.000000 | 0.009661 | 0.021167 |
| 2 | 50 | 2 | 0.856073 | 0.530795 | 0.014475 | -0.065916 | 0.006723 | 0.015006 | 0.030116 |
| 3 | 120 | 2 | 0.839694 | 0.542572 | -0.002377 | -0.079350 | 0.025510 | 0.007312 | 0.016651 |
| 4 | 50 | 3 | 0.838000 | 0.600553 | 0.098619 | 0.019724 | 0.063866 | 0.020911 | 0.042639 |
| 5 | 30 | 2 | 0.827501 | 0.499417 | 0.043808 | -0.076675 | 0.051926 | 0.024807 | 0.043753 |

## Top-5 consistency

- Common configs across all three top-5: [(30, 2), (50, 2), (50, 3), (80, 2)]

## Component Contributions

- contrib_equal.png
- contrib_gating.png
- contrib_regime.png

## Pareto Plots

- pareto_auc_vs_smooth.png
- pareto_corr_vs_smooth.png

## Conclusion Template

We first filtered unstable configurations by smoothness and correlation constraints, then ranked candidates by gating-friendly score. The top configuration balances regime separation (AUC/sep) and prediction consistency (corr_mse) while keeping lambda smoothness within acceptable bounds.
