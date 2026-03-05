## Phase A Summary (Switch-aware)

- Best strategy by directional_align_overall: `score_equal`
- Best strategy by auc_switch_rel: `score_equal`
- Best strategy by retained_gap_switch: `score_gating`
- Main runs pass rate (legacy core checks): `1.000`
- Main runs pass rate (v2 core checks): `0.333`
- Main runs pass rate (v3 core checks): `0.000`
- Main runs pass rate (v3 abs-only): `0.333`
- Negative-control pass rate (v3): `0.000`
- Negative-control drop (directional_align_overall): `0.108906`

### V2 Check Summary
- directional_align_pass: FAIL
- switch_band_pass: FAIL
- pass_core_checks_v2: FAIL

### Notes
- v2 ranking uses switch-aware metrics to better separate true temporal alignment from shuffle/constant/shift controls.

### Top Fail Reasons (v3)
- window_robust_pass: 3
- rel_pass_all: 2
- switch_band_correct_rate_rel_pass: 2
- switch_margin_gap_signed_rel_pass: 2
- peak_delay_min_rel_pass: 2
