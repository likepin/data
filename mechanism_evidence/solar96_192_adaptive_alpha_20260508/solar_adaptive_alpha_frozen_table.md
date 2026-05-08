# Solar Adaptive-Alpha Frozen Table

This table evaluates prediction-level adaptive fusion over existing Solar baseline/static prediction arrays. Validation selects alpha; test is used once for the selected report. No new training is performed.

| horizon | setting | kind | test MSE | test MAE | MSE vs baseline mean | MAE vs baseline mean | MSE vs best single | MAE vs best single |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | global_closed_form_alpha | adaptive_global_alpha | 0.196099 | 0.226875 | +1.576% | +2.215% | +2.633% | +4.355% |
| 96 | per_variable_shrinkage_alpha | adaptive_variable_alpha | 0.196010 | 0.227003 | +1.621% | +2.160% | +2.678% | +4.301% |
| 96 | baseline_mean | group_mean | 0.199239 | 0.232014 | +0.000% | +0.000% | +1.074% | +2.188% |
| 96 | static_mean | group_mean | 0.199067 | 0.226389 | +0.086% | +2.424% | +1.159% | +4.559% |
| 96 | posthoc_closed_loop | guarded_dynamic | 0.204988 | 0.230618 | -2.885% | +0.602% | -1.780% | +2.777% |
| 96 | best_single_baseline_p2 | single_reference | 0.201403 | 0.237205 | -1.086% | -2.237% | +0.000% | +0.000% |
| 192 | global_closed_form_alpha | adaptive_global_alpha | 0.233273 | 0.254255 | +1.009% | +1.671% | +3.843% | +2.794% |
| 192 | per_variable_shrinkage_alpha | adaptive_variable_alpha | 0.233232 | 0.254287 | +1.026% | +1.658% | +3.860% | +2.781% |
| 192 | baseline_mean | group_mean | 0.235650 | 0.258575 | +0.000% | +0.000% | +2.863% | +1.142% |
| 192 | static_mean | group_mean | 0.236968 | 0.255610 | -0.559% | +1.147% | +2.320% | +2.275% |
| 192 | posthoc_closed_loop | guarded_dynamic | 0.243756 | 0.261294 | -3.440% | -1.052% | -0.478% | +0.102% |
| 192 | best_single_static_p1 | single_reference | 0.242596 | 0.261562 | -2.948% | -1.155% | +0.000% | +0.000% |
