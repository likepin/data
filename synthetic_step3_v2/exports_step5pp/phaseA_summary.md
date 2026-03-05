## Phase A Summary (Switch-aware)

- Best strategy by directional_align_overall: `score_gating`
- Best strategy by auc_switch_rel: `score_gating`
- Best strategy by retained_gap_switch: `score_equal`
- Main runs pass rate (legacy core checks): `1.000`
- Main runs pass rate (v2 core checks): `0.000`
- Main runs pass rate (v3 core checks): `0.000`
- Negative-control drop (directional_align_overall): `0.111090`

### V2 Check Summary
- directional_align_pass: FAIL
- switch_band_pass: FAIL
- pass_core_checks_v2: FAIL

### Notes
- v2 ranking uses switch-aware metrics to better separate true temporal alignment from shuffle/constant/shift controls.
