# ETTh1-96 Stage3 Dynamic Oracle Audit

This is a diagnostic upper-bound table. Oracle rows use test labels to decide whether the dynamic correction helps, so they are not reportable method performance.

| split | variant | oracle | MSE | MAE | MSE vs anchor | MAE vs anchor | active ratio | active units |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | static_p0_dynamic | anchor | 0.660592 | 0.537701 | +0.0000% | +0.0000% | 0.0000 | 0 |
| val | static_p0_dynamic | selected_stage3 | 0.653733 | 0.536477 | +1.0384% | +0.2278% | 1.0000 | 2785 |
| val | static_p0_dynamic | oracle_window_gate | 0.652102 | 0.535816 | +1.2852% | +0.3507% | 0.6057 | 1687 |
| val | static_p0_dynamic | oracle_target_gate | 0.650779 | 0.534976 | +1.4855% | +0.5069% | 0.3320 | 6473 |
| val | static_p0_dynamic | oracle_point_gate | 0.648328 | 0.531447 | +1.8565% | +1.1632% | 0.3341 | 625225 |
| test | static_p0_dynamic | anchor | 0.380261 | 0.399500 | +0.0000% | +0.0000% | 0.0000 | 0 |
| test | static_p0_dynamic | selected_stage3 | 0.381877 | 0.399841 | -0.4250% | -0.0855% | 1.0000 | 2785 |
| test | static_p0_dynamic | oracle_window_gate | 0.378003 | 0.397727 | +0.5939% | +0.4438% | 0.4101 | 1142 |
| test | static_p0_dynamic | oracle_target_gate | 0.377228 | 0.396809 | +0.7976% | +0.6734% | 0.3108 | 6060 |
| test | static_p0_dynamic | oracle_point_gate | 0.372907 | 0.390653 | +1.9339% | +2.2144% | 0.3296 | 616800 |
| val | static_mean_dynamic | anchor | 0.660592 | 0.537701 | +0.0000% | +0.0000% | 0.0000 | 0 |
| val | static_mean_dynamic | selected_stage3 | 0.653767 | 0.536511 | +1.0333% | +0.2213% | 1.0000 | 2785 |
| val | static_mean_dynamic | oracle_window_gate | 0.652115 | 0.535825 | +1.2833% | +0.3490% | 0.6086 | 1695 |
| val | static_mean_dynamic | oracle_target_gate | 0.650807 | 0.534999 | +1.4813% | +0.5026% | 0.3309 | 6450 |
| val | static_mean_dynamic | oracle_point_gate | 0.648374 | 0.531482 | +1.8495% | +1.1567% | 0.3330 | 623307 |
| test | static_mean_dynamic | anchor | 0.380261 | 0.399500 | +0.0000% | +0.0000% | 0.0000 | 0 |
| test | static_mean_dynamic | selected_stage3 | 0.381859 | 0.399825 | -0.4201% | -0.0816% | 1.0000 | 2785 |
| test | static_mean_dynamic | oracle_window_gate | 0.378002 | 0.397731 | +0.5940% | +0.4426% | 0.4101 | 1142 |
| test | static_mean_dynamic | oracle_target_gate | 0.377231 | 0.396805 | +0.7968% | +0.6744% | 0.3111 | 6065 |
| test | static_mean_dynamic | oracle_point_gate | 0.372928 | 0.390674 | +1.9284% | +2.2091% | 0.3299 | 617357 |
