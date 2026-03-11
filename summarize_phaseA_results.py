import os
import csv
import json
import argparse
from datetime import datetime

import numpy as np


BENCHMARK_VERSION = "phaseA_v0.1_synth"
SEPARABILITY_CORR_MAX = 0.95
SEPARABILITY_MAD_MIN = 1e-3


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


def find_row(rows, key, value):
    for r in rows:
        if str(r.get(key, "")) == str(value):
            return r
    return None


def metric_str(v):
    fv = to_float(v)
    if np.isnan(fv):
        return "nan"
    return f"{fv:.6f}"


def write_rows_csv(path, rows):
    if not rows:
        return
    header = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                header.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_rows_md(path, rows, columns, title):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"## {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            vals = []
            for col in columns:
                v = row.get(col, "")
                if isinstance(v, bool):
                    vals.append("True" if v else "False")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


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

    benchmark_version = BENCHMARK_VERSION
    generated_at = datetime.now().isoformat(timespec="seconds")
    phaseb_lock_path = os.path.join(exports_dir, "phaseB_locked_variants.json")
    phaseb_lock_payload = read_json(phaseb_lock_path)
    phaseb_locked_variants = {}
    if isinstance(phaseb_lock_payload, dict):
        raw_locked = phaseb_lock_payload.get("locked_variants", {})
        if isinstance(raw_locked, dict):
            phaseb_locked_variants = raw_locked

    current_regime_cfg = find_row(main_cfg, "lambda_strategy", "score_regime") or find_row(main_cfg, "config_name", "score_regime")
    current_regime_check = find_row(main_checks, "lambda_strategy", "score_regime") or find_row(main_checks, "config_name", "score_regime")
    current_regime_vs = find_row(main_vs_rows, "lambda_strategy", "score_regime") or find_row(main_vs_rows, "config_name", "score_regime")
    current_regime_variant = find_row(lambda_variants_rows, "variant_name", "score_regime")
    current_gating_variant = find_row(lambda_variants_rows, "variant_name", "score_gating")

    def build_strategy_snapshot(strategy_name):
        cfg = find_row(main_cfg, "lambda_strategy", strategy_name) or find_row(main_cfg, "config_name", strategy_name) or {}
        chk = find_row(main_checks, "lambda_strategy", strategy_name) or find_row(main_checks, "config_name", strategy_name) or {}
        vs = find_row(main_vs_rows, "lambda_strategy", strategy_name) or find_row(main_vs_rows, "config_name", strategy_name) or {}
        variant = find_row(lambda_variants_rows, "variant_name", strategy_name) or {}
        out = {
            "lambda_strategy": strategy_name,
            "config_name": cfg.get("config_name", chk.get("config_name", strategy_name)),
            "lambda_hash_round6": variant.get("lambda_hash_round6", ""),
            "window": variant.get("window", ""),
            "k": variant.get("k", ""),
            "selected_rank": variant.get("selected_rank", ""),
            "source_csv": variant.get("source_csv", ""),
            "switch_band_correct_rate": to_float(chk.get("switch_band_correct_rate", cfg.get("switch_band_correct_rate"))),
            "switch_margin_gap_signed": to_float(chk.get("switch_margin_gap_signed", cfg.get("switch_margin_gap_signed"))),
            "peak_delay_min": to_float(chk.get("peak_delay_min", cfg.get("peak_delay_min"))),
            "directional_align_overall": to_float(chk.get("directional_align_overall", cfg.get("directional_align_overall"))),
            "delta_align_vs_blockshuffle": to_float(vs.get("delta_align_vs_blockshuffle")),
            "delta_switch_vs_blockshuffle": to_float(vs.get("delta_switch_vs_blockshuffle")),
            "delta_peakdelay_vs_blockshuffle": to_float(vs.get("delta_peakdelay_vs_blockshuffle")),
            "pass_core_checks_v3_v2": to_bool(chk.get("pass_core_checks_v3_v2", cfg.get("pass_core_checks_v3_v2"))),
            "top_fail_reason_v2": chk.get("top_fail_reason_v2", cfg.get("top_fail_reason_v2", "")),
            "fail_reasons_v2": chk.get("fail_reasons_v2", cfg.get("fail_reasons_v2", "")),
        }
        if strategy_name == "score_regime":
            out["corr_gating_regime"] = pair_corr_gr
            out["mean_abs_diff_gating_regime"] = pair_mad_gr
            out["hash_same_gating_regime"] = bool(pair_hash_same_gr)
            out["strategy_collapse"] = bool(strategy_collapse)
            out["negative_control_v3_v2_pass_count"] = neg_v3_v2_pass_count
            out["negative_control_v3_v2_guardrail_pass"] = bool(guardrail_ok_v2)
        return out

    regime_snapshot = build_strategy_snapshot("score_regime")
    gating_snapshot = build_strategy_snapshot("score_gating")

    phaseb_baseline_path = os.path.join(exports_dir, "phaseB_baseline_regime.json")
    baseline_payload = read_json(phaseb_baseline_path)
    if not isinstance(baseline_payload, dict) or not baseline_payload:
        baseline_payload = {
            "benchmark_version": benchmark_version,
            "created_at": generated_at,
            "snapshot": regime_snapshot,
            "note": "Frozen Phase B baseline regime lambda under current synthetic benchmark. Baseline is only promoted after iteration_accept=True.",
        }
        with open(phaseb_baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_payload, f, indent=2)
    baseline_snapshot = baseline_payload.get("snapshot", {}) if isinstance(baseline_payload, dict) else {}
    if not isinstance(baseline_snapshot, dict) or not baseline_snapshot:
        baseline_snapshot = regime_snapshot

    def build_iteration_row(snapshot, baseline, role):
        switch_band = to_float(snapshot.get("switch_band_correct_rate"))
        switch_margin = to_float(snapshot.get("switch_margin_gap_signed"))
        peak_delay = to_float(snapshot.get("peak_delay_min"))
        base_switch_band = to_float(baseline.get("switch_band_correct_rate"))
        base_switch_margin = to_float(baseline.get("switch_margin_gap_signed"))
        base_peak_delay = to_float(baseline.get("peak_delay_min"))
        curr_delta_align = to_float(snapshot.get("delta_align_vs_blockshuffle"))
        curr_delta_switch = to_float(snapshot.get("delta_switch_vs_blockshuffle"))
        curr_delta_peak = to_float(snapshot.get("delta_peakdelay_vs_blockshuffle"))
        base_delta_align = to_float(baseline.get("delta_align_vs_blockshuffle"))
        base_delta_switch = to_float(baseline.get("delta_switch_vs_blockshuffle"))
        base_delta_peak = to_float(baseline.get("delta_peakdelay_vs_blockshuffle"))
        pair_corr = to_float(snapshot.get("corr_gating_regime"))
        pair_mad = to_float(snapshot.get("mean_abs_diff_gating_regime"))
        hash_same = bool(snapshot.get("hash_same_gating_regime"))
        collapse = bool(snapshot.get("strategy_collapse"))
        delta_switch_band = switch_band - base_switch_band if not np.isnan(switch_band) and not np.isnan(base_switch_band) else np.nan
        delta_switch_margin = switch_margin - base_switch_margin if not np.isnan(switch_margin) and not np.isnan(base_switch_margin) else np.nan
        delta_peak_delay = base_peak_delay - peak_delay if not np.isnan(peak_delay) and not np.isnan(base_peak_delay) else np.nan
        delta_align_sep = curr_delta_align - base_delta_align if not np.isnan(curr_delta_align) and not np.isnan(base_delta_align) else np.nan
        delta_switch_sep = curr_delta_switch - base_delta_switch if not np.isnan(curr_delta_switch) and not np.isnan(base_delta_switch) else np.nan
        delta_peak_sep = curr_delta_peak - base_delta_peak if not np.isnan(curr_delta_peak) and not np.isnan(base_delta_peak) else np.nan
        base_corr = to_float(baseline.get("corr_gating_regime"))
        base_mad = to_float(baseline.get("mean_abs_diff_gating_regime"))
        delta_corr = base_corr - pair_corr if not np.isnan(base_corr) and not np.isnan(pair_corr) else np.nan
        delta_mad = pair_mad - base_mad if not np.isnan(base_mad) and not np.isnan(pair_mad) else np.nan
        improved_flags = [
            bool(not np.isnan(delta_switch_band) and delta_switch_band > 0),
            bool(not np.isnan(delta_switch_margin) and delta_switch_margin > 0),
            bool(not np.isnan(delta_peak_delay) and delta_peak_delay > 0),
        ]
        mechanism_improvement_count = int(sum(1 for x in improved_flags if x))
        mechanism_progress = bool(mechanism_improvement_count >= 2) if role != "baseline" else True
        separability_kept = bool(
            (not hash_same) and
            (not collapse) and
            (np.isnan(pair_corr) or pair_corr < SEPARABILITY_CORR_MAX) and
            (np.isnan(pair_mad) or pair_mad > SEPARABILITY_MAD_MIN)
        )
        guardrail_pass = bool(snapshot.get("negative_control_v3_v2_guardrail_pass"))
        pass_core_v3_v2 = bool(snapshot.get("pass_core_checks_v3_v2"))
        if role == "baseline":
            iteration_accept = True
            accept_reason = "baseline_snapshot"
        else:
            reject_reasons = []
            if not pass_core_v3_v2:
                reject_reasons.append("pass_core_checks_v3_v2_false")
            if not guardrail_pass:
                reject_reasons.append("guardrail_fail")
            if not separability_kept:
                reject_reasons.append("separability_lost")
            if not mechanism_progress:
                reject_reasons.append("mechanism_progress_lt_2of3")
            iteration_accept = bool(len(reject_reasons) == 0)
            accept_reason = "accepted" if iteration_accept else ";".join(reject_reasons)
        iteration_key = "|".join([
            benchmark_version,
            str(snapshot.get("lambda_hash_round6", "")),
            str(gating_snapshot.get("lambda_hash_round6", "")),
            metric_str(switch_band),
            metric_str(switch_margin),
            metric_str(peak_delay),
            metric_str(pair_corr),
            metric_str(pair_mad),
        ])
        return {
            "iteration_key": iteration_key,
            "timestamp": generated_at,
            "benchmark_version": benchmark_version,
            "iteration_role": role,
            "regime_baseline_hash_used": baseline.get("lambda_hash_round6", ""),
            "regime_baseline_window_used": baseline.get("window", ""),
            "regime_baseline_k_used": baseline.get("k", ""),
            "regime_lambda_strategy": snapshot.get("lambda_strategy", "score_regime"),
            "regime_config_name": snapshot.get("config_name", "score_regime"),
            "regime_lambda_hash_round6": snapshot.get("lambda_hash_round6", ""),
            "regime_window": snapshot.get("window", ""),
            "regime_k": snapshot.get("k", ""),
            "regime_selected_rank": snapshot.get("selected_rank", ""),
            "regime_source_csv": snapshot.get("source_csv", ""),
            "gating_lambda_hash_round6": gating_snapshot.get("lambda_hash_round6", ""),
            "gating_window": gating_snapshot.get("window", ""),
            "gating_k": gating_snapshot.get("k", ""),
            "switch_band_correct_rate": switch_band,
            "delta_switch_band_vs_baseline": delta_switch_band,
            "switch_margin_gap_signed": switch_margin,
            "delta_switch_margin_gap_vs_baseline": delta_switch_margin,
            "peak_delay_min": peak_delay,
            "delta_peak_delay_vs_baseline": delta_peak_delay,
            "directional_align_overall": to_float(snapshot.get("directional_align_overall")),
            "delta_align_vs_blockshuffle": curr_delta_align,
            "delta_switch_vs_blockshuffle": curr_delta_switch,
            "delta_peakdelay_vs_blockshuffle": curr_delta_peak,
            "delta_align_vs_blockshuffle_vs_baseline": delta_align_sep,
            "delta_switch_vs_blockshuffle_vs_baseline": delta_switch_sep,
            "delta_peakdelay_vs_blockshuffle_vs_baseline": delta_peak_sep,
            "corr_gating_regime": pair_corr,
            "delta_corr_vs_baseline": delta_corr,
            "mean_abs_diff_gating_regime": pair_mad,
            "delta_mad_vs_baseline": delta_mad,
            "hash_same_gating_regime": hash_same,
            "strategy_collapse": collapse,
            "negative_control_v3_v2_pass_count": snapshot.get("negative_control_v3_v2_pass_count", neg_v3_v2_pass_count),
            "negative_control_v3_v2_guardrail_pass": guardrail_pass,
            "pass_core_checks_v3_v2": pass_core_v3_v2,
            "mechanism_improvement_count": mechanism_improvement_count,
            "mechanism_progress": mechanism_progress,
            "separability_kept": separability_kept,
            "top_fail_reason_v2": snapshot.get("top_fail_reason_v2", ""),
            "fail_reasons_v2": snapshot.get("fail_reasons_v2", ""),
            "iteration_accept": iteration_accept,
            "iteration_accept_reason": accept_reason,
        }

    iteration_csv = os.path.join(compare_dir, "compare_regime_iteration.csv")
    iteration_md = os.path.join(compare_dir, "compare_regime_iteration.md")
    iteration_rows = read_csv(iteration_csv)
    existing_by_key = {
        str(r.get("iteration_key", "")): idx
        for idx, r in enumerate(iteration_rows)
        if r.get("iteration_key")
    }
    baseline_row = build_iteration_row(baseline_snapshot, baseline_snapshot, role="baseline")
    if baseline_row["iteration_key"] not in existing_by_key:
        baseline_row["iteration_id"] = len(iteration_rows)
        iteration_rows.append(baseline_row)
        existing_by_key[baseline_row["iteration_key"]] = len(iteration_rows) - 1
    else:
        keep_id = iteration_rows[existing_by_key[baseline_row["iteration_key"]]].get("iteration_id")
        baseline_row["iteration_id"] = keep_id
        iteration_rows[existing_by_key[baseline_row["iteration_key"]]] = baseline_row
    current_role = "baseline" if str(regime_snapshot.get("lambda_hash_round6", "")) == str(baseline_snapshot.get("lambda_hash_round6", "")) else "candidate"
    current_iteration_row = build_iteration_row(regime_snapshot, baseline_snapshot, role=current_role)
    if current_iteration_row["iteration_key"] not in existing_by_key:
        current_iteration_row["iteration_id"] = len(iteration_rows)
        iteration_rows.append(current_iteration_row)
        existing_by_key[current_iteration_row["iteration_key"]] = len(iteration_rows) - 1
    else:
        keep_idx = existing_by_key[current_iteration_row["iteration_key"]]
        keep_id = iteration_rows[keep_idx].get("iteration_id")
        current_iteration_row["iteration_id"] = keep_id
        iteration_rows[keep_idx] = current_iteration_row

    dedup_rows = []
    seen_signatures = {}
    for row in iteration_rows:
        sig = "|".join([
            str(row.get("benchmark_version", "")),
            str(row.get("iteration_role", "")),
            str(row.get("regime_lambda_hash_round6", "")),
            str(row.get("gating_lambda_hash_round6", "")),
        ])
        if sig in seen_signatures:
            dedup_rows[seen_signatures[sig]] = row
        else:
            seen_signatures[sig] = len(dedup_rows)
            dedup_rows.append(row)
    iteration_rows = dedup_rows

    for idx, row in enumerate(iteration_rows):
        row["iteration_id"] = idx

    baseline_promoted_this_run = False
    if current_role == "candidate" and bool(current_iteration_row.get("iteration_accept")):
        baseline_payload = {
            "benchmark_version": benchmark_version,
            "created_at": baseline_payload.get("created_at", generated_at),
            "updated_at": generated_at,
            "promoted_from_iteration_key": current_iteration_row.get("iteration_key", ""),
            "snapshot": regime_snapshot,
            "note": "Frozen Phase B baseline regime lambda under current synthetic benchmark. Baseline is only promoted after iteration_accept=True.",
        }
        with open(phaseb_baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_payload, f, indent=2)
        baseline_promoted_this_run = True

    iteration_md_cols = [
        "iteration_id",
        "timestamp",
        "iteration_role",
        "benchmark_version",
        "regime_baseline_hash_used",
        "regime_lambda_hash_round6",
        "switch_band_correct_rate",
        "delta_switch_band_vs_baseline",
        "switch_margin_gap_signed",
        "delta_switch_margin_gap_vs_baseline",
        "peak_delay_min",
        "delta_peak_delay_vs_baseline",
        "delta_switch_vs_blockshuffle",
        "delta_switch_vs_blockshuffle_vs_baseline",
        "corr_gating_regime",
        "delta_corr_vs_baseline",
        "mean_abs_diff_gating_regime",
        "delta_mad_vs_baseline",
        "hash_same_gating_regime",
        "strategy_collapse",
        "negative_control_v3_v2_guardrail_pass",
        "pass_core_checks_v3_v2",
        "iteration_accept",
        "iteration_accept_reason",
    ]
    write_rows_csv(iteration_csv, iteration_rows)
    write_rows_md(iteration_md, iteration_rows, iteration_md_cols, title="Regime Iteration Ledger")

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
        "benchmark_version": benchmark_version,
        "phaseB_baseline_regime_path": phaseb_baseline_path,
        "phaseB_locked_variants_path": phaseb_lock_path,
        "phaseB_locked_variants_present": bool(phaseb_locked_variants),
        "phaseB_baseline_promoted_this_run": baseline_promoted_this_run,
        "compare_regime_iteration_csv": iteration_csv,
        "compare_regime_iteration_md": iteration_md,
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
        f.write(f"- Benchmark version: `{benchmark_version}`\n")
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
        f.write(f"- Phase B baseline snapshot: `{os.path.basename(phaseb_baseline_path)}`\n")
        f.write(f"- Phase B locked equal/gating: `{os.path.basename(phaseb_lock_path)}` present=`{bool(phaseb_locked_variants)}`\n")
        f.write(f"- Phase B iteration ledger: `{os.path.basename(iteration_csv)}` / `{os.path.basename(iteration_md)}`\n")
        f.write(f"- Phase B baseline promoted this run: `{baseline_promoted_this_run}`\n")
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
        f.write(f"- Benchmark version: `{benchmark_version}`\n")
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
        f.write(f"- Phase B baseline snapshot: `{os.path.basename(phaseb_baseline_path)}`\n")
        f.write(f"- Phase B iteration ledger: `{os.path.basename(iteration_csv)}` / `{os.path.basename(iteration_md)}`\n")
        f.write("- Recommendation: keep this rule as the provisional synthetic benchmark standard and re-validate before transferring to real data.\n")

    print(f"[OK] {out_json}")
    print(f"[OK] {out_md}")
    print(f"[OK] {out_block_md}")
    print(f"[OK] {out_rulebook_md}")
    print(f"[OK] {phaseb_baseline_path}")
    print(f"[OK] {iteration_csv}")
    print(f"[OK] {iteration_md}")


if __name__ == "__main__":
    main()
