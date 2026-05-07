# Traffic96 Stage2-Light Seed2026 Evidence Package

This package freezes the light Stage2 Traffic adaptive-alpha performance branch. It stores only small CSV/JSON/log/figure artifacts and intentionally excludes large `.npy` arrays.

- Selected test MSE/MAE: `0.382640 / 0.259420`
- Gain vs static_p1: MSE `+2.4370%`, MAE `+3.3201%`
- Increment vs Stage1.5 selected: MSE `+0.0782%`, MAE `+0.0949%`

Main subpackage: `performance/adaptive_alpha_ensemble/`.
