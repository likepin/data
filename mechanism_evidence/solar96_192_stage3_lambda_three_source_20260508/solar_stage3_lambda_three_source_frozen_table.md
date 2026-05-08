# Solar Stage3 Lambda Three-Source Frozen Table

Stage3 adds a validation-selected lambda-gated dynamic increment on top of the Solar adaptive-alpha baseline/static anchor. Test is used only for the selected rows.

| horizon | label | variant | test MSE | test MAE | MSE vs adaptive | MAE vs adaptive | MSE vs baseline mean | MAE vs baseline mean | selected |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 96 | Stage3 closed-form eta2 | static_mean_dynamic | 0.195924 | 0.226947 | +0.0437% | +0.0245% | +1.6638% | +2.1837% | stage3_closed_form_top_alpha_5 |
| 96 | Stage3 closed-form eta2 | static_p0_dynamic | 0.195922 | 0.226946 | +0.0451% | +0.0250% | +1.6652% | +2.1841% | stage3_closed_form_top_alpha_5 |
| 96 | adaptive-alpha anchor | anchor | 0.196010 | 0.227003 | +0.0000% | +0.0000% | +1.6208% | +2.1597% | per_variable_shrinkage_alpha |
| 192 | Stage3 closed-form eta2 | static_mean_dynamic | 0.233232 | 0.254287 | +0.0000% | +0.0000% | +1.0261% | +1.6584% | stage2_anchor |
| 192 | Stage3 closed-form eta2 | static_p0_dynamic | 0.233232 | 0.254287 | +0.0000% | +0.0000% | +1.0261% | +1.6584% | stage2_anchor |
| 192 | adaptive-alpha anchor | anchor | 0.233232 | 0.254287 | +0.0000% | +0.0000% | +1.0261% | +1.6584% | per_variable_shrinkage_alpha |
