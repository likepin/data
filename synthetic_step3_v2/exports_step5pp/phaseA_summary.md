## Phase A Summary (Switch-aware)

- Benchmark version: `phaseA_v0.1_synth`
- Best strategy by directional_align_overall: `score_equal`
- Best strategy by auc_switch_rel: `score_regime`
- Best strategy by retained_gap_switch: `score_regime`
- Main runs pass rate (legacy core checks): `1.000`
- Main runs pass rate (v2 core checks): `0.000`
- Main runs pass rate (v3 core checks): `0.000`
- Main runs pass rate (v3 before guardrail): `0.000`
- Main runs pass rate (v3 abs-only): `0.000`
- Main runs pass rate (v3_v2 core checks): `0.000`
- Main runs pass rate (v3_v2 before guardrail): `0.000`
- Main runs pass rate (v3_v2 abs-only): `0.000`
- Negative-control pass rate (v3): `0.000`
- Negative-control v3 pass count: `0` / max `1` (PASS)
- Negative-control pass rate (v3_v2): `0.000`
- Negative-control v3_v2 pass count: `0` / max `1` (PASS)
- corr(score_gating, score_regime): `0.569042`
- mean_abs_diff(score_gating, score_regime): `2.183536e-01`
- strategy_collapse: `False`
- Negative-control drop (directional_align_overall): `0.176969`
- peak_delay_min mean (main): `66.333333`
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
- Phase B locked equal/gating: `phaseB_locked_variants.json` present=`True`
- Phase B iteration ledger: `compare_regime_iteration.csv` / `compare_regime_iteration.md`
- Phase B baseline promoted this run: `False`

### V2 Check Summary
- directional_align_pass: FAIL
- switch_band_pass: FAIL
- pass_core_checks_v2: FAIL

### V3_v2 Window Fail Breakdown (failed main rows)
- window_100_core_abs_fail_count_v2: 3
- window_200_core_abs_fail_count_v2: 2
- window_400_core_abs_fail_count_v2: 1

### Notes
- v2 ranking uses switch-aware metrics to better separate true temporal alignment from shuffle/constant/shift controls.

### Top Fail Reasons (v3_v2)
- switch_margin_gap_signed_abs_pass_v2: 2
- switch_margin_gap_signed_rel_pass_v2: 2
- window_200_core_abs_pass_v2: 2
- window_robust_pass_v2: 2
- peak_delay_min_abs_pass_v2: 1
- peak_delay_min_rel_pass_v2: 1
- switch_band_correct_rate_abs_pass_v2: 1
- switch_band_correct_rate_rel_pass_v2: 1
