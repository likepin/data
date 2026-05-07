# Traffic96 Stage2-Light Frozen Performance Table

Stage2-Light adds one paired seed (`projection_3`, seed=2026) to the existing three baseline/staticcausal projections. Selection remains validation-only. The Stage1.5 selected row is included only as a historical reference.

| setting | kind | alpha | val MSE | val MAE | test MSE | test MAE | test MSE gain vs static_p1 | test MAE gain vs static_p1 | test MSE gain vs Stage1.5 | test MAE gain vs Stage1.5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage1.5 selected per-variable alpha | previous_reference | mean=0.668060; std=0.148481 | 0.350266 | 0.239480 | 0.382939 | 0.259666 | +2.3606% | +3.2283% | +0.0000% | +0.0000% |
| best single static_p1 | single_reference |  | 0.357625 | 0.248371 | 0.392198 | 0.268328 | +0.0000% | +0.0000% | -2.4177% | -3.3360% |
| baseline mean, 4 seeds | group_mean | 0.00 static | 0.354987 | 0.241954 | 0.387347 | 0.262069 | +1.2370% | +2.3329% | -1.1508% | -0.9252% |
| staticcausal mean, 4 seeds | group_mean | 1.00 static | 0.351671 | 0.240388 | 0.384510 | 0.260429 | +1.9602% | +2.9440% | -0.4101% | -0.2938% |
| alpha=0.50 equal blend | global_blend | 0.50 | 0.351710 | 0.239439 | 0.384065 | 0.259553 | +2.0735% | +3.2703% | -0.2940% | +0.0435% |
| global closed-form alpha | adaptive_global_alpha | 0.755954 | 0.351285 | 0.239494 | 0.383828 | 0.259580 | +2.1342% | +3.2602% | -0.2319% | +0.0330% |
| per-variable shrinkage alpha | adaptive_variable_alpha | mean=0.784892; std=0.144948 | 0.349884 | 0.239219 | 0.382640 | 0.259420 | +2.4370% | +3.3201% | +0.0782% | +0.0949% |
