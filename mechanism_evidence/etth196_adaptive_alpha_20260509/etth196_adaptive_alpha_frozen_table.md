# ETTh1-96 Adaptive-Alpha Frozen Table

This table evaluates prediction-level adaptive fusion over existing ETTh1 baseline/static anchor predictions. Validation selects alpha; test is used once for the selected report. No new training is performed.

| setting | kind | alpha | val MSE | val MAE | test MSE | test MAE | test MSE vs baseline mean | test MAE vs baseline mean | test MSE vs static mean | test MAE vs static mean | test MSE vs best single | test MAE vs best single |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_mean | group_mean | 0.00 static | 0.668775 | 0.540585 | 0.381221 | 0.400097 | +0.000% | +0.000% | +0.265% | +0.260% | +2.022% | +1.609% |
| static_mean | group_mean | 1.00 static | 0.664627 | 0.540230 | 0.382232 | 0.401138 | -0.265% | -0.260% | +0.000% | +0.000% | +1.762% | +1.352% |
| best_single_static_p1 | single_reference |  | 0.671656 | 0.544096 | 0.389088 | 0.406638 | -2.063% | -1.635% | -1.793% | -1.371% | +0.000% | +0.000% |
| global_closed_form_alpha | adaptive_global_alpha | 0.645142 | 0.662828 | 0.538585 | 0.379996 | 0.399264 | +0.321% | +0.208% | +0.585% | +0.467% | +2.337% | +1.813% |
| per_variable_shrinkage_alpha | adaptive_variable_alpha | mean=0.615778; std=0.191568 | 0.660592 | 0.537701 | 0.380261 | 0.399500 | +0.252% | +0.149% | +0.516% | +0.409% | +2.269% | +1.755% |
| posthoc_closed_loop | guarded_dynamic | Selective:selective_activation |  |  | 0.388088 | 0.405779 | -1.801% | -1.420% | -1.532% | -1.157% | +0.257% | +0.211% |
