import os
import csv
import json
import argparse

import numpy as np


def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "y")


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def argmax_row(rows, key):
    best = None
    best_v = -np.inf
    for r in rows:
        v = to_float(r.get(key))
        if np.isnan(v):
            continue
        if v > best_v:
            best_v = v
            best = r
    return best, best_v


def mean_valid(rows, key):
    vals = [to_float(r.get(key)) for r in rows]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def pass_rate(rows, key):
    if not rows:
        return np.nan
    vals = [1.0 if to_bool(r.get(key)) else 0.0 for r in rows]
    return float(np.mean(vals)) if vals else np.nan


def find_pair_row(rows, a, b):
    for r in rows:
        x = str(r.get("variant_a", ""))
        y = str(r.get("variant_b", ""))
        if (x == a and y == b) or (x == b and y == a):
            return r
    return None


def semicolon_items(text):
    if text is None:
        return []
    return [x.strip() for x in str(text).split(";") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--exports_dir", type=str, default=None)
    parser.add_argument("--compare_dir", type=str, default=None)
    args = parser.parse_args()

    exports_dir = args.exports_dir or os.path.join(args.data_dir, "exports_step5pp")
    compare_dir = args.compare_dir or os.path.join(exports_dir, "compare")

    cfg_rows = read_csv(os.path.join(compare_dir, "compare_phaseA_configs.csv"))
    check_rows = read_csv(os.path.join(compare_dir, "compare_phaseA_checks.csv"))
    main_vs_rows = read_csv(os.path.join(compare_dir, "compare_phaseA_main_vs_blockshuffle.csv"))
    block_rows = read_csv(os.path.join(compare_dir, "compare_phaseA_blockshuffle.csv"))
    lambda_variants_rows = read_csv(os.path.join(exports_dir, "lambda_variants_summary.csv"))
    if not lambda_variants_rows:
        js = read_json(os.path.join(exports_dir, "lambda_variants_summary.json"))
        if isinstance(js, list):
            lambda_variants_rows = js
    lambda_pairwise_rows = read_csv(os.path.join(exports_dir, "lambda_pairwise.csv"))
    if not lambda_pairwise_rows:
        js = read_json(os.path.join(exports_dir, "lambda_pairwise.json"))
        if isinstance(js, list):
            lambda_pairwise_rows = js
    if not cfg_rows:
        raise RuntimeError("compare_phaseA_configs.csv missing or empty.")
    if not check_rows:
        raise RuntimeError("compare_phaseA_checks.csv missing or empty.")

    main_cfg = [r for r in cfg_rows if str(r.get("run_type", "")).lower() == "main"] or cfg_rows
    neg_cfg = [r for r in cfg_rows if str(r.get("run_type", "")).lower() == "negative_control"]
    main_checks = [r for r in check_rows if str(r.get("run_type", "")).lower() == "main"] or check_rows
    neg_checks = [r for r in check_rows if str(r.get("run_type", "")).lower() == "negative_control"]

    best_dir_row, _ = argmax_row(main_cfg, "directional_align_overall")
    best_auc_row, _ = argmax_row(main_cfg, "auc_switch_rel")
    best_gap_row, _ = argmax_row(main_cfg, "retained_gap_switch")
    best_by_directional = best_dir_row.get("lambda_strategy") if best_dir_row else None
    best_by_auc = best_auc_row.get("lambda_strategy") if best_auc_row else None
    best_by_gap_switch = best_gap_row.get("lambda_strategy") if best_gap_row else None

    pass_v1_vals = [1.0 if to_bool(r.get("pass_core_checks")) else 0.0 for r in main_checks]
    pass_v2_vals = [1.0 if to_bool(r.get("pass_core_checks_v2")) else 0.0 for r in main_checks]
    pass_v3_vals = [1.0 if to_bool(r.get("pass_core_checks_v3")) else 0.0 for r in main_checks]
    pass_v3_raw_vals = [1.0 if to_bool(r.get("pass_core_checks_v3_before_guardrail")) else 0.0 for r in main_checks]
    pass_v3_v2_vals = [1.0 if to_bool(r.get("pass_core_checks_v3_v2")) else 0.0 for r in main_checks]
    pass_v3_v2_raw_vals = [1.0 if to_bool(r.get("pass_core_checks_v3_v2_before_guardrail")) else 0.0 for r in main_checks]
    main_pass_rate_v1 = float(np.mean(pass_v1_vals)) if pass_v1_vals else np.nan
    main_pass_rate_v2 = float(np.mean(pass_v2_vals)) if pass_v2_vals else np.nan
    main_pass_rate_v3 = float(np.mean(pass_v3_vals)) if pass_v3_vals else np.nan
    main_pass_rate_v3_before_guardrail = float(np.mean(pass_v3_raw_vals)) if pass_v3_raw_vals else np.nan
    main_pass_rate_v3_v2 = float(np.mean(pass_v3_v2_vals)) if pass_v3_v2_vals else np.nan
    main_pass_rate_v3_v2_before_guardrail = float(np.mean(pass_v3_v2_raw_vals)) if pass_v3_v2_raw_vals else np.nan

    main_directional_mean = mean_valid(main_checks, "directional_align_overall")
    neg_directional_mean = mean_valid(neg_checks, "directional_align_overall")
    main_v3_abs_rate = pass_rate(main_checks, "pass_core_checks_v3_abs")
    main_v3_abs_rate_v2 = pass_rate(main_checks, "pass_core_checks_v3_abs_v2")
    neg_v3_rate = pass_rate(neg_checks, "pass_core_checks_v3")
    neg_v3_rate_v2 = pass_rate(neg_checks, "pass_core_checks_v3_v2")
    neg_v3_pass_count = int(sum(1 for r in neg_checks if to_bool(r.get("pass_core_checks_v3"))))
    neg_v3_v2_pass_count = int(sum(1 for r in neg_checks if to_bool(r.get("pass_core_checks_v3_v2"))))
    guardrail_max_allowed_raw = to_float((main_checks or neg_checks or [{}])[0].get("negative_control_v3_pass_max_allowed"))
    guardrail_max_allowed = int(guardrail_max_allowed_raw) if np.isfinite(guardrail_max_allowed_raw) else 1
    if guardrail_max_allowed <= 0:
        guardrail_max_allowed = 1
    guardrail_ok = bool(neg_v3_pass_count <= guardrail_max_allowed)
    guardrail_max_allowed_v2_raw = to_float((main_checks or neg_checks or [{}])[0].get("negative_control_v3_v2_pass_max_allowed"))
    guardrail_max_allowed_v2 = int(guardrail_max_allowed_v2_raw) if np.isfinite(guardrail_max_allowed_v2_raw) else 1
    if guardrail_max_allowed_v2 <= 0:
        guardrail_max_allowed_v2 = 1
    guardrail_ok_v2 = bool(neg_v3_v2_pass_count <= guardrail_max_allowed_v2)

    fail_counts_v3_v2 = {}
    for r in main_checks:
        if to_bool(r.get("pass_core_checks_v3_v2")):
            continue
        for item in semicolon_items(r.get("fail_reasons_v2")):
            fail_counts_v3_v2[item] = fail_counts_v3_v2.get(item, 0) + 1
    fail_rank_v3_v2 = sorted(fail_counts_v3_v2.items(), key=lambda x: x[1], reverse=True)
    top_fail_reasons_v3_v2 = [{"key": k, "count": int(v)} for k, v in fail_rank_v3_v2[:8]]

    failed_main_rows = [r for r in main_checks if not to_bool(r.get("pass_core_checks_v3_v2"))]
    window_fail_breakdown = {
        "window_100_core_abs_fail_count_v2": int(sum(1 for r in failed_main_rows if not to_bool(r.get("window_100_core_abs_pass_v2")))),
        "window_200_core_abs_fail_count_v2": int(sum(1 for r in failed_main_rows if not to_bool(r.get("window_200_core_abs_pass_v2")))),
        "window_400_core_abs_fail_count_v2": int(sum(1 for r in failed_main_rows if not to_bool(r.get("window_400_core_abs_pass_v2")))),
    }
    if np.isfinite(main_directional_mean) and np.isfinite(neg_directional_mean):
        negative_control_drop_v2 = float(main_directional_mean - neg_directional_mean)
    else:
        negative_control_drop_v2 = np.nan

    shift_rows = [r for r in neg_checks if str(r.get("control_family", "")).lower() == "shift"]
    block_shuffle_rows = [r for r in neg_checks if str(r.get("control_family", "")).lower() == "block_shuffle"]
    main_peak_delay_mean = mean_valid(main_checks, "peak_delay_min")
    shift_peak_delay_mean = mean_valid(shift_rows, "peak_delay_min")
    block_peak_delay_mean = mean_valid(block_shuffle_rows, "peak_delay_min")
    peak_delay_abs_thr_v2 = to_float((main_checks or check_rows or [{}])[0].get("peak_delay_min_abs_thr_v2"))
    peak_delay_rel_thr_v2 = to_float((main_checks or check_rows or [{}])[0].get("peak_delay_min_rel_thr_v2"))
    peak_delay_abs_rule_v2 = str((main_checks or check_rows or [{}])[0].get("peak_delay_min_abs_rule_v2", "") or "")
    peak_delay_rel_rule_v2 = str((main_checks or check_rows or [{}])[0].get("peak_delay_min_rel_rule_v2", "") or "")

    pair_gr = find_pair_row(lambda_pairwise_rows, "score_gating", "score_regime")
    pair_corr_gr = to_float(pair_gr.get("corr")) if pair_gr else np.nan
    pair_mad_gr = to_float(pair_gr.get("mean_abs_diff")) if pair_gr else np.nan
    pair_hash_same_gr = to_bool(pair_gr.get("hash_same")) if pair_gr else False
    collapse_reasons = []
    if pair_hash_same_gr:
        collapse_reasons.append("hash_same")
    if np.isfinite(pair_corr_gr) and pair_corr_gr > 0.99:
        collapse_reasons.append("corr_gt_0.99")
    if np.isfinite(pair_mad_gr) and pair_mad_gr < 1e-3:
        collapse_reasons.append("mean_abs_diff_lt_1e-3")
    strategy_collapse = bool(len(collapse_reasons) > 0)
    if not pair_gr:
        collapse_reasons.append("pairwise_missing")
    collapse_reason = ";".join(collapse_reasons)

    summary_json = {
        # legacy-compatible keys
        "best_strategy_by_align": best_by_directional,
        "best_strategy_by_retained_gap": best_by_gap_switch,
        "main_runs_pass_rate": main_pass_rate_v1,
        "negative_control_drop": negative_control_drop_v2,
        # v2 keys
        "best_strategy_by_directional_align": best_by_directional,
        "best_strategy_by_switch_auc": best_by_auc,
        "best_strategy_by_retained_gap_switch": best_by_gap_switch,
        "main_runs_pass_rate_v1": main_pass_rate_v1,
        "main_runs_pass_rate_v2": main_pass_rate_v2,
        "main_runs_pass_rate_v3": main_pass_rate_v3,
        "main_runs_pass_rate_v3_before_guardrail": main_pass_rate_v3_before_guardrail,
        "main_runs_pass_rate_v3_abs": main_v3_abs_rate,
        "main_runs_pass_rate_v3_v2": main_pass_rate_v3_v2,
        "main_runs_pass_rate_v3_v2_before_guardrail": main_pass_rate_v3_v2_before_guardrail,
        "main_runs_pass_rate_v3_abs_v2": main_v3_abs_rate_v2,
        "negative_control_pass_rate_v3": neg_v3_rate,
        "negative_control_pass_rate_v3_v2": neg_v3_rate_v2,
        "negative_control_v3_pass_count": neg_v3_pass_count,
        "negative_control_v3_pass_max_allowed": guardrail_max_allowed,
        "negative_control_v3_guardrail_pass": guardrail_ok,
        "negative_control_v3_v2_pass_count": neg_v3_v2_pass_count,
        "negative_control_v3_v2_pass_max_allowed": guardrail_max_allowed_v2,
        "negative_control_v3_v2_guardrail_pass": guardrail_ok_v2,
        "negative_control_drop_v2": negative_control_drop_v2,
        "main_directional_align_mean": main_directional_mean,
        "neg_directional_align_mean": neg_directional_mean,
        "main_peak_delay_min_mean": main_peak_delay_mean,
        "shift_peak_delay_min_mean": shift_peak_delay_mean,
        "block_shuffle_peak_delay_min_mean": block_peak_delay_mean,
        "provisional_phaseA_rule": {
            "legacy_fields_retained": True,
            "peak_delay_min_abs_thr_v2": peak_delay_abs_thr_v2,
            "peak_delay_min_rel_thr_v2": peak_delay_rel_thr_v2,
            "peak_delay_min_abs_rule_v2": peak_delay_abs_rule_v2,
            "peak_delay_min_rel_rule_v2": peak_delay_rel_rule_v2,
            "note": "Current synthetic PhaseA provisional standard. Re-validate before using on real data.",
        },
        "lambda_pair_corr_score_gating_score_regime": pair_corr_gr,
        "lambda_pair_mean_abs_diff_score_gating_score_regime": pair_mad_gr,
        "lambda_pair_hash_same_score_gating_score_regime": bool(pair_hash_same_gr),
        "strategy_collapse": strategy_collapse,
        "collapse_reason": collapse_reason,
        "window_fail_breakdown_v3_v2": window_fail_breakdown,
        "top_fail_reasons_v3_v2": top_fail_reasons_v3_v2,
    }

    out_json = os.path.join(exports_dir, "phaseA_summary.json")
    out_md = os.path.join(exports_dir, "phaseA_summary.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    directional_pass = all(to_bool(r.get("directional_align_pass")) for r in main_checks) if main_checks else False
    switch_band_pass = all(to_bool(r.get("switch_band_pass")) for r in main_checks) if main_checks else False
    v2_pass_all = all(to_bool(r.get("pass_core_checks_v2")) for r in main_checks) if main_checks else False

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("## Phase A Summary (Switch-aware)\n\n")
        f.write(f"- Best strategy by directional_align_overall: `{best_by_directional}`\n")
        f.write(f"- Best strategy by auc_switch_rel: `{best_by_auc}`\n")
        f.write(f"- Best strategy by retained_gap_switch: `{best_by_gap_switch}`\n")
        f.write(f"- Main runs pass rate (legacy core checks): `{main_pass_rate_v1:.3f}`\n")
        f.write(f"- Main runs pass rate (v2 core checks): `{main_pass_rate_v2:.3f}`\n")
        f.write(f"- Main runs pass rate (v3 core checks): `{main_pass_rate_v3:.3f}`\n")
        f.write(f"- Main runs pass rate (v3 before guardrail): `{main_pass_rate_v3_before_guardrail:.3f}`\n")
        f.write(f"- Main runs pass rate (v3 abs-only): `{main_v3_abs_rate:.3f}`\n")
        f.write(f"- Main runs pass rate (v3_v2 core checks): `{main_pass_rate_v3_v2:.3f}`\n")
        f.write(f"- Main runs pass rate (v3_v2 before guardrail): `{main_pass_rate_v3_v2_before_guardrail:.3f}`\n")
        f.write(f"- Main runs pass rate (v3_v2 abs-only): `{main_v3_abs_rate_v2:.3f}`\n")
        f.write(f"- Negative-control pass rate (v3): `{neg_v3_rate:.3f}`\n")
        f.write(f"- Negative-control v3 pass count: `{neg_v3_pass_count}` / max `{guardrail_max_allowed}` ({'PASS' if guardrail_ok else 'FAIL'})\n")
        f.write(f"- Negative-control pass rate (v3_v2): `{neg_v3_rate_v2:.3f}`\n")
        f.write(f"- Negative-control v3_v2 pass count: `{neg_v3_v2_pass_count}` / max `{guardrail_max_allowed_v2}` ({'PASS' if guardrail_ok_v2 else 'FAIL'})\n")
        if np.isfinite(pair_corr_gr):
            f.write(f"- corr(score_gating, score_regime): `{pair_corr_gr:.6f}`\n")
        else:
            f.write("- corr(score_gating, score_regime): `nan`\n")
        if np.isfinite(pair_mad_gr):
            f.write(f"- mean_abs_diff(score_gating, score_regime): `{pair_mad_gr:.6e}`\n")
        else:
            f.write("- mean_abs_diff(score_gating, score_regime): `nan`\n")
        f.write(f"- strategy_collapse: `{strategy_collapse}`\n")
        if collapse_reason:
            f.write(f"- collapse_reason: `{collapse_reason}`\n")
        if not np.isnan(negative_control_drop_v2):
            f.write(f"- Negative-control drop (directional_align_overall): `{negative_control_drop_v2:.6f}`\n")
        else:
            f.write("- Negative-control drop (directional_align_overall): `nan`\n")
        if np.isfinite(main_peak_delay_mean):
            f.write(f"- peak_delay_min mean (main): `{main_peak_delay_mean:.6f}`\n")
        if np.isfinite(shift_peak_delay_mean):
            f.write(f"- peak_delay_min mean (shift): `{shift_peak_delay_mean:.6f}`\n")
        if np.isfinite(block_peak_delay_mean):
            f.write(f"- peak_delay_min mean (block_shuffle): `{block_peak_delay_mean:.6f}`\n")
        f.write("\n")
        f.write("### Provisional PhaseA Rule\n")
        if np.isfinite(peak_delay_abs_thr_v2):
            f.write(f"- peak_delay_min_abs_thr_v2: `{peak_delay_abs_thr_v2:.6f}`\n")
        else:
            f.write("- peak_delay_min_abs_thr_v2: `nan`\n")
        if peak_delay_abs_rule_v2:
            f.write(f"- peak_delay_min_abs_rule_v2: `{peak_delay_abs_rule_v2}`\n")
        if np.isfinite(peak_delay_rel_thr_v2):
            f.write(f"- peak_delay_min_rel_thr_v2: `{peak_delay_rel_thr_v2:.6f}`\n")
        else:
            f.write("- peak_delay_min_rel_thr_v2: `nan`\n")
        if peak_delay_rel_rule_v2:
            f.write(f"- peak_delay_min_rel_rule_v2: `{peak_delay_rel_rule_v2}`\n")
        f.write("- legacy v3/v2 fields are retained for backward compatibility.\n")
        f.write("- This is the current synthetic PhaseA provisional standard, not a universal threshold.\n")
        f.write("\n")
        f.write("### V2 Check Summary\n")
        f.write(f"- directional_align_pass: {'PASS' if directional_pass else 'FAIL'}\n")
        f.write(f"- switch_band_pass: {'PASS' if switch_band_pass else 'FAIL'}\n")
        f.write(f"- pass_core_checks_v2: {'PASS' if v2_pass_all else 'FAIL'}\n")
        f.write("\n")
        f.write("### V3_v2 Window Fail Breakdown (failed main rows)\n")
        f.write(f"- window_100_core_abs_fail_count_v2: {window_fail_breakdown['window_100_core_abs_fail_count_v2']}\n")
        f.write(f"- window_200_core_abs_fail_count_v2: {window_fail_breakdown['window_200_core_abs_fail_count_v2']}\n")
        f.write(f"- window_400_core_abs_fail_count_v2: {window_fail_breakdown['window_400_core_abs_fail_count_v2']}\n")
        f.write("\n")
        f.write("### Notes\n")
        f.write("- v2 ranking uses switch-aware metrics to better separate true temporal alignment from shuffle/constant/shift controls.\n")
        if not guardrail_ok:
            f.write("- Guardrail triggered: too many negative controls passed v3; all v3 passes are invalidated.\n")
        if not guardrail_ok_v2:
            f.write("- Guardrail triggered: too many negative controls passed v3_v2; all v3_v2 passes are invalidated.\n")
        if top_fail_reasons_v3_v2:
            f.write("\n### Top Fail Reasons (v3_v2)\n")
            for r in top_fail_reasons_v3_v2:
                f.write(f"- {r['key']}: {r['count']}\n")

    out_block_md = os.path.join(exports_dir, "phaseA_blockshuffle_summary.md")
    out_rulebook_md = os.path.join(exports_dir, "phaseA_rulebook.md")
    if main_vs_rows and block_rows:
        block_align_mean = mean_valid(block_rows, "directional_align_overall")
        block_switch_mean = mean_valid(block_rows, "switch_band_correct_rate")
        block_peak_mean = mean_valid(block_rows, "peak_delay_lambda")
        best_main, _ = argmax_row(main_vs_rows, "delta_switch_vs_blockshuffle")
        with open(out_block_md, "w", encoding="utf-8") as f:
            f.write("## Phase A Block-Shuffle Summary\n\n")
            f.write(f"- blockshuffle mean directional_align_overall: `{block_align_mean:.6f}`\n")
            f.write(f"- blockshuffle mean switch_band_correct_rate: `{block_switch_mean:.6f}`\n")
            f.write(f"- blockshuffle mean peak_delay_lambda: `{block_peak_mean:.6f}`\n")
            if best_main:
                f.write(f"- best main strategy vs blockshuffle (delta_switch): `{best_main.get('lambda_strategy')}`\n")
            f.write("\n")
            f.write("### Main vs Blockshuffle\n")
            f.write("| config_name | lambda_strategy | delta_align_vs_blockshuffle | delta_switch_vs_blockshuffle | delta_peakdelay_vs_blockshuffle | pass_core_checks_v3 | pass_core_checks_v3_v2 |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for r in main_vs_rows:
                f.write(
                    f"| {r.get('config_name')} | {r.get('lambda_strategy')} | {r.get('delta_align_vs_blockshuffle')} | "
                    f"{r.get('delta_switch_vs_blockshuffle')} | {r.get('delta_peakdelay_vs_blockshuffle')} | {r.get('pass_core_checks_v3')} | {r.get('pass_core_checks_v3_v2')} |\n"
                )
    else:
        with open(out_block_md, "w", encoding="utf-8") as f:
            f.write("## Phase A Block-Shuffle Summary\n\n")
            f.write("- Missing compare_phaseA_main_vs_blockshuffle.csv or compare_phaseA_blockshuffle.csv.\n")

    with open(out_rulebook_md, "w", encoding="utf-8") as f:
        f.write("## PhaseA Rulebook\n\n")
        f.write("This document records the current provisional evaluation rule used for the synthetic PhaseA benchmark.\n\n")
        f.write("### Rule Status\n")
        f.write("- Scope: synthetic PhaseA only\n")
        f.write("- Legacy fields retained: `True`\n")
        f.write("- Purpose: recover valid main strategies without letting negative controls pass\n\n")
        f.write("### Peak Delay v3_v2\n")
        if np.isfinite(peak_delay_abs_thr_v2):
            f.write(f"- peak_delay_min_abs_thr_v2: `{peak_delay_abs_thr_v2:.6f}`\n")
        else:
            f.write("- peak_delay_min_abs_thr_v2: `nan`\n")
        if peak_delay_abs_rule_v2:
            f.write(f"- peak_delay_min_abs_rule_v2: `{peak_delay_abs_rule_v2}`\n")
        if np.isfinite(peak_delay_rel_thr_v2):
            f.write(f"- peak_delay_min_rel_thr_v2: `{peak_delay_rel_thr_v2:.6f}`\n")
        else:
            f.write("- peak_delay_min_rel_thr_v2: `nan`\n")
        if peak_delay_rel_rule_v2:
            f.write(f"- peak_delay_min_rel_rule_v2: `{peak_delay_rel_rule_v2}`\n")
        f.write("- Interpretation: peak delay is treated as a temporal misalignment diagnostic and is calibrated against `shift` controls.\n\n")
        f.write("### Current Outcome\n")
        f.write(f"- main_runs_pass_rate_v3_v2: `{main_pass_rate_v3_v2:.3f}`\n")
        f.write(f"- negative_control_pass_rate_v3_v2: `{neg_v3_rate_v2:.3f}`\n")
        f.write("- Recommendation: keep this rule as the provisional synthetic benchmark standard and re-validate before transferring to real data.\n")

    print(f"[OK] {out_json}")
    print(f"[OK] {out_md}")
    print(f"[OK] {out_block_md}")
    print(f"[OK] {out_rulebook_md}")


if __name__ == "__main__":
    main()
