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
