# Traffic96 Stage1.5 Frozen Performance Table

All deterministic settings are evaluated on validation and test from existing prediction arrays. The shuffled-alpha row is a target-identity negative control and reports median MSE across shuffles.

| setting | kind | alpha | val MSE | val MAE | val MSE gain | val MAE gain | test MSE | test MAE | test MSE gain | test MAE gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best single static_p1 | single_reference |  | 0.357625 | 0.248371 | +0.0000% | +0.0000% | 0.392198 | 0.268328 | +0.0000% | +0.0000% |
| baseline mean | group_mean | 0.00 static | 0.354398 | 0.242034 | +0.9022% | +2.5512% | 0.387362 | 0.262332 | +1.2331% | +2.2348% |
| staticcausal mean | group_mean | 1.00 static | 0.352535 | 0.241407 | +1.4234% | +2.8038% | 0.385032 | 0.261319 | +1.8272% | +2.6123% |
| alpha=0.50 equal blend | global_blend | 0.50 | 0.351589 | 0.239563 | +1.6876% | +3.5463% | 0.384028 | 0.259724 | +2.0830% | +3.2068% |
| alpha=0.60 grid selected | global_blend | 0.60 | 0.351478 | 0.239588 | +1.7188% | +3.5363% | 0.383882 | 0.259708 | +2.1203% | +3.2126% |
| global closed-form alpha | adaptive_global_alpha | 0.624119 | 0.351474 | 0.239620 | +1.7200% | +3.5234% | 0.383873 | 0.259730 | +2.1226% | +3.2045% |
| per-variable shrinkage alpha | adaptive_variable_alpha | mean=0.668060; std=0.148481 | 0.350266 | 0.239480 | +2.0577% | +3.5794% | 0.382939 | 0.259666 | +2.3606% | +3.2283% |
| shuffled alpha median | negative_control | 256 shuffles | 0.351656 |  | +1.6692% |  | 0.384043 |  | +2.0792% |  |
