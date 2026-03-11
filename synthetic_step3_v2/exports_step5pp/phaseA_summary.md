## Phase A Summary (Switch-aware)

- Benchmark version: `phaseA_v0.1_synth`
- Best strategy by directional_align_overall: `score_equal`
- Best strategy by auc_switch_rel: `score_gating`
- Best strategy by retained_gap_switch: `score_gating`
- Main runs pass rate (legacy core checks): `1.000`
- Main runs pass rate (v2 core checks): `0.000`
- Main runs pass rate (v3 core checks): `0.333`
- Main runs pass rate (v3 before guardrail): `0.333`
- Main runs pass rate (v3 abs-only): `0.333`
- Main runs pass rate (v3_v2 core checks): `0.333`
- Main runs pass rate (v3_v2 before guardrail): `0.333`
- Main runs pass rate (v3_v2 abs-only): `0.333`
- Negative-control pass rate (v3): `0.000`
- Negative-control v3 pass count: `0` / max `1` (PASS)
- Negative-control pass rate (v3_v2): `0.000`
- Negative-control v3_v2 pass count: `0` / max `1` (PASS)
- corr(score_gating, score_regime): `0.840015`
- mean_abs_diff(score_gating, score_regime): `1.201030e-01`
- strategy_collapse: `False`
- Negative-control drop (directional_align_overall): `0.188711`
- peak_delay_min mean (main): `75.666667`
- peak_delay_min mean (shift): `100.250000`
- peak_delay_min mean (block_shuffle): `75.750000`

### Provisional PhaseA Rule
- peak_delay_min_abs_thr_v2: `130.000000`
- peak_delay_min_abs_rule_v2: `min(0.65*switch_window,max(default_abs_thr,mapped_shift_q75))`
- peak_delay_min_rel_thr_v2: `130.500000`
- peak_delay_min_rel_rule_v2: `value <= mapped_shift_q75`
- legacy v3/v2 fields are retained for backward compatibility.
- This is the current synthetic PhaseA provisional standard, not a universal threshold.
- Phase B baseline snapshot: `phaseB_baseline_regime.json`
- Phase B iteration ledger: `compare_regime_iteration.csv` / `compare_regime_iteration.md`

### V2 Check Summary
- directional_align_pass: FAIL
- switch_band_pass: PASS
- pass_core_checks_v2: FAIL

### V3_v2 Window Fail Breakdown (failed main rows)
- window_100_core_abs_fail_count_v2: 2
- window_200_core_abs_fail_count_v2: 1
- window_400_core_abs_fail_count_v2: 0

### Notes
- v2 ranking uses switch-aware metrics to better separate true temporal alignment from shuffle/constant/shift controls.

### Top Fail Reasons (v3_v2)
- switch_margin_gap_signed_abs_pass_v2: 1
- switch_margin_gap_signed_rel_pass_v2: 1
- window_200_core_abs_pass_v2: 1
- window_robust_pass_v2: 1
- peak_delay_min_abs_pass_v2: 1
- peak_delay_min_rel_pass_v2: 1
