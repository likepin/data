# ETTh1-96 Stage3 Lambda Three-Source Frozen Table

Stage3 adds a validation-selected lambda-gated dynamic increment on top of the ETTh1 adaptive-alpha anchor. This is a post-hoc dynamic add-on, not a new training run.

| label | variant | test MSE | test MAE | MSE vs adaptive | MAE vs adaptive | MSE vs baseline mean | MAE vs baseline mean | selected | eta | eta_raw | clip | target_mask |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| Stage3 closed-form eta2 | static_mean_dynamic | 0.381859 | 0.399825 | -0.4201% | -0.0816% | -0.1672% | +0.0679% | stage3_closed_form_all | 2.000 | 8.904 | clipped_high | all |
| Stage3 closed-form eta2 | static_p0_dynamic | 0.381877 | 0.399841 | -0.4250% | -0.0855% | -0.1721% | +0.0639% | stage3_closed_form_all | 2.000 | 8.970 | clipped_high | all |
| adaptive-alpha anchor | anchor | 0.380261 | 0.399500 | +0.0000% | +0.0000% | +0.2519% | +0.1493% | per_variable_shrinkage_alpha | 0.000 | 0.000 | anchor | n/a |
