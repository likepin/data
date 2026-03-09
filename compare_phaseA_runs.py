import os
import csv
import json
import argparse
import shutil
import re

import numpy as np


def warn(msg):
    print(f"WARN: {msg}")


def read_json_or_none(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_md_table(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip().startswith("|")]
    if len(lines) < 3:
        return rows
    header = [x.strip() for x in lines[0].strip("|").split("|")]
    for ln in lines[2:]:
        vals = [x.strip() for x in ln.strip("|").split("|")]
        if len(vals) != len(header):
            continue
        rows.append({header[i]: vals[i] for i in range(len(header))})
    return rows


def read_rows(csv_path, md_path):
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return parse_md_table(md_path)


def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def pick_subset(rows, subset):
    for r in rows:
        if r.get("subset") == subset:
            return r
    return None


def infer_run_type(run_name):
    name = run_name.lower()
    if "shuffle" in name or "constant" in name or "const" in name or "shift" in name:
        return "negative_control"
    return "main"


def infer_control_family(run_name):
    name = run_name.lower()
    if "block_shuffle" in name:
        return "block_shuffle"
    if "shift" in name:
        return "shift"
    if "constant" in name or "const" in name:
        return "constant"
    if "shuffle" in name:
        return "shuffle_global"
    return "main"


def infer_control_seed(run_name):
    m = re.search(r"_s(\d+)$", str(run_name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def pick_metric(sanity, checks, key, default=np.nan):
    if key in sanity and sanity.get(key) is not None:
        return sanity.get(key)
    if key in checks and checks.get(key) is not None:
        return checks.get(key)
    return default


def normalize_cell(v):
    if v is None:
        return "NaN"
    if isinstance(v, (bool, np.bool_)):
        return "True" if bool(v) else "False"
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "NaN"
    return v


def finite_list(values):
    out = []
    for v in values:
        fv = to_float(v)
        if not np.isnan(fv):
            out.append(fv)
    return out


def row_peak_delay_min(row):
    vals = finite_list([
        row.get("peak_delay_lambda"),
        row.get("peak_delay_gate"),
        row.get("peak_delay_rel"),
        row.get("peak_delay_min"),
    ])
    return float(min(vals)) if vals else np.nan


def semicolon_join(parts):
    return ";".join([str(p) for p in parts if str(p).strip()])


def ensure_signed_abs_fields(row):
    sp = to_float(row.get("switch_margin_pre_signed", row.get("switch_margin_pre")))
    so = to_float(row.get("switch_margin_post_signed", row.get("switch_margin_post")))
    rg = to_float(row.get("retained_gap_switch_signed", row.get("retained_gap_switch")))
    if not np.isnan(sp) and not np.isnan(so):
        sg = float(min(sp, so))
    else:
        sg = np.nan
    row["switch_margin_pre_signed"] = sp
    row["switch_margin_post_signed"] = so
    row["switch_margin_pre_abs"] = float(abs(sp)) if not np.isnan(sp) else np.nan
    row["switch_margin_post_abs"] = float(abs(so)) if not np.isnan(so) else np.nan
    row["switch_margin_gap_signed"] = sg
    row["switch_margin_gap_abs"] = float(abs(sg)) if not np.isnan(sg) else np.nan
    row["retained_gap_switch_signed"] = rg
    row["retained_gap_switch_abs"] = float(abs(rg)) if not np.isnan(rg) else np.nan
    row["peak_delay_min"] = row_peak_delay_min(row)
    return row


def metric_value_oriented(row, spec):
    metric = spec["name"]
    better = spec["better"]
    v = to_float(row.get(metric))
    if np.isnan(v):
        return np.nan
    if better == "lower":
        return -v
    return v


def load_window_checks(run_dir, base_window):
    out = {}
    base = read_json_or_none(os.path.join(run_dir, "checks.json")) or {}
    bw = to_float(base.get("switch_window", base_window))
    if np.isnan(bw):
        bw = float(base_window)
    out[int(bw)] = base
    for w in (100, 200, 400):
        p = os.path.join(run_dir, f"checks_sw{w}.json")
        js = read_json_or_none(p)
        if js:
            out[int(w)] = js
    return out


METRIC_SPECS = [
    {
        "name": "directional_align_overall",
        "better": "higher",
        "abs_thr_default": 0.58,
        "control_family_for_rel": "block_shuffle",
        "abs_required": True,
        "rel_required": True,
        "window_vote": True,
    },
    {
        "name": "switch_band_correct_rate",
        "better": "higher",
        "abs_thr_default": 0.60,
        "control_family_for_rel": "block_shuffle",
        "abs_required": True,
        "rel_required": True,
        "window_vote": True,
    },
    {
        "name": "switch_margin_gap_signed",
        "better": "higher",
        "abs_thr_default": 0.0,
        "control_family_for_rel": "block_shuffle",
        "abs_required": True,
        "rel_required": True,
        "window_vote": True,
    },
    {
        "name": "peak_delay_min",
        "better": "lower",
        "abs_thr_default": np.nan,  # threshold uses switch_window
        "control_family_for_rel": "shift",
        "abs_required": True,
        "rel_required": True,
        "window_vote": False,
    },
    {
        "name": "retained_gap_switch_abs",
        "better": "higher",
        "abs_thr_default": 0.10,
        "control_family_for_rel": "block_shuffle",
        "abs_required": True,
        "rel_required": False,
        "window_vote": False,
    },
    {
        "name": "retained_gap_switch_signed",
        "better": "lower",
        "abs_thr_default": -0.10,
        "control_family_for_rel": "block_shuffle",
        "abs_required": False,
        "rel_required": False,
        "window_vote": False,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--exports_dir", type=str, default=None)
    parser.add_argument("--runs_dir", type=str, default=None)
    parser.add_argument("--compare_dir", type=str, default=None)
    args = parser.parse_args()

    exports_dir = args.exports_dir or os.path.join(args.data_dir, "exports_step5pp")
    runs_dir = args.runs_dir or os.path.join(exports_dir, "runs")
    compare_dir = args.compare_dir or os.path.join(exports_dir, "compare")
    os.makedirs(compare_dir, exist_ok=True)

    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    config_rows = []
    subset_rows_out = []
    check_rows = []
    run_allowlist = None
    status_path = os.path.join(compare_dir, "batch_run_status.json")
    status_rows = read_json_or_none(status_path)
    if isinstance(status_rows, list) and status_rows:
        run_allowlist = set([str(r.get("run_name")) for r in status_rows if r.get("run_name")])

    for run_name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        if run_allowlist is not None and run_name not in run_allowlist:
            continue
        sanity = read_json_or_none(os.path.join(run_dir, "sanity_metrics.json")) or {}
        checks = read_json_or_none(os.path.join(run_dir, "checks.json")) or {}
        config_used = read_json_or_none(os.path.join(run_dir, "config_used.json")) or {}
        subset_rows = read_rows(
            os.path.join(run_dir, "subset_summary.csv"),
            os.path.join(run_dir, "subset_summary.md"),
        )
        if not subset_rows:
            # backward compatibility fallback
            subset_rows = read_rows(
                os.path.join(run_dir, "step5pp_summary.csv"),
                os.path.join(run_dir, "step5pp_summary.md"),
            )
            if subset_rows:
                warn(f"{run_name}: using fallback step5pp_summary.*")

        high_row = pick_subset(subset_rows, "high_non_sat") or pick_subset(subset_rows, "high_sat")
        low_row = pick_subset(subset_rows, "low")
        retained_gap = np.nan
        if high_row and low_row:
            retained_gap = to_float(low_row.get("mean_retained_ratio")) - to_float(high_row.get("mean_retained_ratio"))
        retained_gap_switch = to_float(sanity.get("retained_gap_switch", checks.get("retained_gap_switch")))

        lambda_strategy = sanity.get("config_name") or config_used.get("lambda_tag") or run_name
        run_type = config_used.get("run_type") or infer_run_type(run_name)
        control_family = config_used.get("control_family") or infer_control_family(run_name)
        control_seed = config_used.get("control_seed")
        if control_seed in ("", None):
            control_seed = infer_control_seed(run_name)

        gate_direction = to_bool(checks.get("gate_direction", sanity.get("gate_direction")))
        high_closer = to_bool(checks.get("high_closer_A0", sanity.get("high_closer_A0")))
        low_closer = to_bool(checks.get("low_closer_A1", sanity.get("low_closer_A1")))
        pass_core = bool(gate_direction and high_closer and low_closer)
        pass_core_v2 = to_bool(checks.get("pass_core_checks_v2", sanity.get("pass_core_checks_v2")))
        if pass_core_v2 is None:
            pass_core_v2 = False
        pass_core_v3 = to_bool(checks.get("pass_core_checks_v3", sanity.get("pass_core_checks_v3")))
        if pass_core_v3 is None:
            pass_core_v3 = False
        pass_core_v3_abs = to_bool(checks.get("pass_core_checks_v3_abs", sanity.get("pass_core_checks_v3_abs")))
        if pass_core_v3_abs is None:
            pass_core_v3_abs = False
        switch_band_pass = to_bool(pick_metric(sanity, checks, "switch_band_pass"))
        directional_align_pass = to_bool(pick_metric(sanity, checks, "directional_align_pass"))
        switch_margin_pass = to_bool(pick_metric(sanity, checks, "switch_margin_pass"))
        peak_delay_pass = to_bool(pick_metric(sanity, checks, "peak_delay_pass"))
        retained_gap_switch_pass = to_bool(pick_metric(sanity, checks, "retained_gap_switch_pass"))

        cfg_row = {
                "config_name": run_name,
                "lambda_strategy": lambda_strategy,
                "run_type": run_type,
                "control_family": control_family,
                "control_seed": control_seed,
                "delta_mask_mode": sanity.get("delta_mask_mode", config_used.get("delta_mask_mode")),
                "dist_mask_mode": sanity.get("dist_mask_mode", config_used.get("dist_mask_mode")),
                "delta_mask_nnz": sanity.get("delta_mask_nnz", config_used.get("delta_mask_nnz")),
                "dist_mask_nnz": sanity.get("dist_mask_nnz", config_used.get("dist_mask_nnz")),
                "align_pre": sanity.get("align_pre", sanity.get("align_all_pre")),
                "align_post": sanity.get("align_post", sanity.get("align_all_post")),
                "align_overall": sanity.get("align_overall", sanity.get("overall_align")),
                "margin_pre": sanity.get("margin_pre", sanity.get("mean_margin_pre")),
                "margin_post": sanity.get("margin_post", sanity.get("mean_margin_post")),
                "dist_std_base": sanity.get("dist_std_base", config_used.get("dist_std_base")),
                "dist_std_reg0": sanity.get("dist_std_reg0", config_used.get("dist_std_reg0")),
                "dist_std_reg1": sanity.get("dist_std_reg1", config_used.get("dist_std_reg1")),
                "gate_direction": gate_direction,
                "high_closer_A0": high_closer,
                "low_closer_A1": low_closer,
                "regime_swapped": sanity.get("regime_swapped", config_used.get("regime_swapped")),
                "swap_reason": sanity.get("swap_reason", config_used.get("swap_reason")),
                "pre_correct_rate": pick_metric(sanity, checks, "pre_correct_rate"),
                "post_correct_rate": pick_metric(sanity, checks, "post_correct_rate"),
                "directional_align_pre": pick_metric(sanity, checks, "directional_align_pre", pick_metric(sanity, checks, "pre_correct_rate")),
                "directional_align_post": pick_metric(sanity, checks, "directional_align_post", pick_metric(sanity, checks, "post_correct_rate")),
                "directional_align_overall": pick_metric(sanity, checks, "directional_align_overall"),
                "switch_window": pick_metric(sanity, checks, "switch_window"),
                "switch_pre_correct_rate": pick_metric(sanity, checks, "switch_pre_correct_rate"),
                "switch_post_correct_rate": pick_metric(sanity, checks, "switch_post_correct_rate"),
                "switch_band_correct_rate": pick_metric(sanity, checks, "switch_band_correct_rate"),
                "switch_margin_pre": pick_metric(sanity, checks, "switch_margin_pre"),
                "switch_margin_post": pick_metric(sanity, checks, "switch_margin_post"),
                "corr_lambda_regime": pick_metric(sanity, checks, "corr_lambda_regime"),
                "corr_gate_regime": pick_metric(sanity, checks, "corr_gate_regime"),
                "corr_retained_regime": pick_metric(sanity, checks, "corr_retained_regime"),
                "auc_switch_lambda": pick_metric(sanity, checks, "auc_switch_lambda"),
                "auc_switch_gate": pick_metric(sanity, checks, "auc_switch_gate"),
                "auc_switch_rel": pick_metric(sanity, checks, "auc_switch_rel"),
                "peak_delay_lambda": pick_metric(sanity, checks, "peak_delay_lambda"),
                "peak_delay_gate": pick_metric(sanity, checks, "peak_delay_gate"),
                "peak_delay_rel": pick_metric(sanity, checks, "peak_delay_rel"),
                "corr_time_lambda_switch": pick_metric(sanity, checks, "corr_time_lambda_switch"),
                "corr_time_gate_switch": pick_metric(sanity, checks, "corr_time_gate_switch"),
                "corr_time_retained_switch": pick_metric(sanity, checks, "corr_time_retained_switch"),
                "retained_gap": to_float(sanity.get("retained_gap", retained_gap)),
                "retained_gap_switch": retained_gap_switch,
                "regime_swapped": sanity.get("regime_swapped", config_used.get("regime_swapped")),
                "swap_reason": sanity.get("swap_reason", config_used.get("swap_reason")),
                "switch_band_pass": bool(switch_band_pass),
                "directional_align_pass": bool(directional_align_pass),
                "switch_margin_pass": bool(switch_margin_pass),
                "peak_delay_pass": bool(peak_delay_pass),
                "retained_gap_switch_pass": bool(retained_gap_switch_pass),
                "pass_core_checks_v2": pass_core_v2,
                "pass_core_checks_v3_abs": pass_core_v3_abs,
                "pass_core_checks_v3": pass_core_v3,
                "_run_dir": run_dir,
            }
        ensure_signed_abs_fields(cfg_row)
        config_rows.append(cfg_row)

        for r in subset_rows:
            subset_rows_out.append(
                {
                    "lambda_strategy": lambda_strategy,
                    "run_type": run_type,
                    "control_family": control_family,
                    "subset": r.get("subset"),
                    "count": r.get("count"),
                    "mean_lambda": r.get("mean_lambda"),
                    "mean_gate_weight": r.get("mean_gate_weight"),
                    "mean_dist_base": r.get("mean_dist_base"),
                    "mean_dist_reg0": r.get("mean_dist_reg0"),
                    "mean_dist_reg1": r.get("mean_dist_reg1"),
                    "mean_retained_ratio": r.get("mean_retained_ratio"),
                }
            )

        check_row = {
                "lambda_strategy": lambda_strategy,
                "config_name": run_name,
                "run_type": run_type,
                "control_family": control_family,
                "control_seed": control_seed,
                "gate_direction": bool(gate_direction),
                "high_closer_A0": bool(high_closer),
                "low_closer_A1": bool(low_closer),
                "regime_swapped": sanity.get("regime_swapped", config_used.get("regime_swapped")),
                "swap_reason": sanity.get("swap_reason", config_used.get("swap_reason")),
                "align_overall": sanity.get("align_overall", sanity.get("overall_align")),
                "margin_pre": sanity.get("margin_pre", sanity.get("mean_margin_pre")),
                "margin_post": sanity.get("margin_post", sanity.get("mean_margin_post")),
                "retained_gap": retained_gap,
                "pre_correct_rate": pick_metric(sanity, checks, "pre_correct_rate"),
                "post_correct_rate": pick_metric(sanity, checks, "post_correct_rate"),
                "directional_align_pre": pick_metric(sanity, checks, "directional_align_pre", pick_metric(sanity, checks, "pre_correct_rate")),
                "directional_align_post": pick_metric(sanity, checks, "directional_align_post", pick_metric(sanity, checks, "post_correct_rate")),
                "directional_align_overall": pick_metric(sanity, checks, "directional_align_overall"),
                "switch_window": pick_metric(sanity, checks, "switch_window"),
                "switch_pre_correct_rate": pick_metric(sanity, checks, "switch_pre_correct_rate"),
                "switch_post_correct_rate": pick_metric(sanity, checks, "switch_post_correct_rate"),
                "switch_band_correct_rate": pick_metric(sanity, checks, "switch_band_correct_rate"),
                "switch_margin_pre": pick_metric(sanity, checks, "switch_margin_pre"),
                "switch_margin_post": pick_metric(sanity, checks, "switch_margin_post"),
                "corr_lambda_regime": pick_metric(sanity, checks, "corr_lambda_regime"),
                "corr_gate_regime": pick_metric(sanity, checks, "corr_gate_regime"),
                "corr_retained_regime": pick_metric(sanity, checks, "corr_retained_regime"),
                "auc_switch_lambda": pick_metric(sanity, checks, "auc_switch_lambda"),
                "auc_switch_gate": pick_metric(sanity, checks, "auc_switch_gate"),
                "auc_switch_rel": pick_metric(sanity, checks, "auc_switch_rel"),
                "peak_delay_lambda": pick_metric(sanity, checks, "peak_delay_lambda"),
                "peak_delay_gate": pick_metric(sanity, checks, "peak_delay_gate"),
                "peak_delay_rel": pick_metric(sanity, checks, "peak_delay_rel"),
                "corr_time_lambda_switch": pick_metric(sanity, checks, "corr_time_lambda_switch"),
                "corr_time_gate_switch": pick_metric(sanity, checks, "corr_time_gate_switch"),
                "corr_time_retained_switch": pick_metric(sanity, checks, "corr_time_retained_switch"),
                "retained_gap_switch": retained_gap_switch,
                "switch_margin_pre_signed": pick_metric(sanity, checks, "switch_margin_pre_signed", pick_metric(sanity, checks, "switch_margin_pre")),
                "switch_margin_post_signed": pick_metric(sanity, checks, "switch_margin_post_signed", pick_metric(sanity, checks, "switch_margin_post")),
                "retained_gap_switch_signed": pick_metric(sanity, checks, "retained_gap_switch_signed", retained_gap_switch),
                "peak_delay_min": pick_metric(sanity, checks, "peak_delay_min"),
                "switch_band_pass": bool(switch_band_pass),
                "directional_align_pass": bool(directional_align_pass),
                "switch_margin_pass": bool(switch_margin_pass),
                "peak_delay_pass": bool(peak_delay_pass),
                "retained_gap_switch_pass": bool(retained_gap_switch_pass),
                "pass_core_checks": pass_core,
                "pass_core_checks_v2": pass_core_v2,
                "pass_core_checks_v3_abs": pass_core_v3_abs,
                "pass_core_checks_v3": pass_core_v3,
                "_run_dir": run_dir,
            }
        ensure_signed_abs_fields(check_row)
        check_rows.append(check_row)

    if not config_rows:
        raise RuntimeError("No valid run results found in runs_dir.")

    # P0/P1: build control thresholds with metric->family mapping and medium strictness.
    def calc_stats(values):
        vals = [to_float(v) for v in values]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "q10": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "q90": np.nan,
                "p95": np.nan,
            }
        arr = np.array(vals, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "q10": float(np.quantile(arr, 0.10)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "q90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
        }

    def medium_rel_thr(stat, better):
        m = to_float(stat.get("mean"))
        s = to_float(stat.get("std"))
        q25 = to_float(stat.get("q25"))
        q75 = to_float(stat.get("q75"))
        if np.isnan(m) or np.isnan(s):
            return np.nan
        if better == "higher":
            if np.isnan(q75):
                return m + 0.5 * s
            return max(m + 0.5 * s, q75)
        if np.isnan(q25):
            return m - 0.5 * s
        return min(m - 0.5 * s, q25)

    neg_rows = [r for r in config_rows if str(r.get("run_type", "")).lower() == "negative_control"]
    known_families = sorted(set(str(r.get("control_family", "")).lower() for r in neg_rows if r.get("control_family")))
    thresholds = {
        "control_pool": "metric_mapped_family",
        "rel_rule_medium": {
            "higher": "value >= max(mean + 0.5*std, q75)",
            "lower": "value <= min(mean - 0.5*std, q25)",
        },
        "abs_rule_v2": {
            "directional_align_overall": "min(default_abs_thr, max(0.52, mapped_q75 + 0.10, all_negative_q90 + 0.01))",
            "others": "default_abs_thr",
        },
        "metrics": {},
        "per_family": {},
        "all_negative_controls": {"n": len(neg_rows), "metrics": {}},
    }

    for fam in known_families:
        fam_rows = [r for r in neg_rows if str(r.get("control_family", "")).lower() == fam]
        fam_node = {"n": len(fam_rows), "metrics": {}}
        for spec in METRIC_SPECS:
            metric = spec["name"]
            stat = calc_stats([r.get(metric) for r in fam_rows])
            stat["better"] = spec["better"]
            stat["thr_medium"] = medium_rel_thr(stat, spec["better"])
            fam_node["metrics"][metric] = stat
        thresholds["per_family"][fam] = fam_node

    for spec in METRIC_SPECS:
        metric = spec["name"]
        stat = calc_stats([r.get(metric) for r in neg_rows])
        stat["better"] = spec["better"]
        stat["thr_medium"] = medium_rel_thr(stat, spec["better"])
        thresholds["all_negative_controls"]["metrics"][metric] = stat

    for spec in METRIC_SPECS:
        metric = spec["name"]
        fam = str(spec["control_family_for_rel"]).lower()
        fam_rows = [r for r in neg_rows if str(r.get("control_family", "")).lower() == fam]
        control_family_used = fam
        fallback = None
        if not fam_rows:
            fam_rows = neg_rows
            control_family_used = "all_negative_controls"
            fallback = fam
        stat = calc_stats([r.get(metric) for r in fam_rows])
        stat["better"] = spec["better"]
        stat["control_family"] = fam
        stat["control_family_used"] = control_family_used
        stat["fallback_from_family"] = fallback
        stat["thr_medium"] = medium_rel_thr(stat, spec["better"])
        thresholds["metrics"][metric] = stat
    thresholds_json = os.path.join(compare_dir, "phaseA_thresholds.json")
    with open(thresholds_json, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    # P0-3 + P1-2: dual v3 criteria (abs + relative-to-control + window robustness).
    def abs_threshold_for(spec, row):
        if spec["name"] == "peak_delay_min":
            sw = to_float(row.get("switch_window"))
            if np.isnan(sw):
                sw = 200.0
            return 0.5 * sw
        return float(spec["abs_thr_default"])

    def abs_pass_for(spec, value, abs_thr):
        if np.isnan(value) or np.isnan(abs_thr):
            return False
        if spec["better"] == "higher":
            return bool(value >= abs_thr)
        return bool(value <= abs_thr)

    def abs_threshold_for_v2(spec, row):
        default_thr = abs_threshold_for(spec, row)
        if spec["name"] != "directional_align_overall":
            return default_thr, "default_abs_thr"
        mapped_stat = thresholds["metrics"].get(spec["name"], {})
        all_neg_stat = thresholds["all_negative_controls"]["metrics"].get(spec["name"], {})
        mapped_q75 = to_float(mapped_stat.get("q75"))
        all_neg_q90 = to_float(all_neg_stat.get("q90"))
        candidates = [0.52]
        if not np.isnan(mapped_q75):
            candidates.append(mapped_q75 + 0.10)
        if not np.isnan(all_neg_q90):
            candidates.append(all_neg_q90 + 0.01)
        thr = float(min(default_thr, max(candidates))) if not np.isnan(default_thr) else float(max(candidates))
        return thr, "min(default_abs_thr,max(0.52,mapped_q75+0.10,all_negative_q90+0.01))"

    def relative_pass_for(spec, value):
        stat = thresholds["metrics"].get(spec["name"], {})
        better = spec["better"]
        m = to_float(stat.get("mean"))
        s = to_float(stat.get("std"))
        q10 = to_float(stat.get("q10"))
        q25 = to_float(stat.get("q25"))
        q75 = to_float(stat.get("q75"))
        q90 = to_float(stat.get("q90"))
        p95 = to_float(stat.get("p95"))
        rel_thr = to_float(stat.get("thr_medium"))
        if np.isnan(value) or np.isnan(m) or np.isnan(rel_thr):
            return False, m, s, q10, q25, q75, q90, p95, rel_thr, np.nan
        if better == "higher":
            rel_pass = bool(value >= rel_thr)
            margin = value - m
        else:
            rel_pass = bool(value <= rel_thr)
            margin = m - value
        return rel_pass, m, s, q10, q25, q75, q90, p95, rel_thr, margin

    for row in config_rows:
        abs_passes = []
        rel_passes = []
        abs_passes_v2 = []
        rel_passes_v2 = []
        fail_flags_v2 = []
        for spec in METRIC_SPECS:
            name = spec["name"]
            value = to_float(row.get(name))
            value_oriented = metric_value_oriented(row, spec)
            abs_thr = abs_threshold_for(spec, row)
            abs_pass = abs_pass_for(spec, value, abs_thr)
            abs_thr_v2, abs_rule_v2 = abs_threshold_for_v2(spec, row)
            abs_pass_v2 = abs_pass_for(spec, value, abs_thr_v2)
            rel_pass, ctrl_mean, ctrl_std, ctrl_q10, ctrl_q25, ctrl_q75, ctrl_q90, ctrl_p95, rel_thr, margin_ctrl = relative_pass_for(spec, value)
            key = name
            row[f"{key}_better"] = spec["better"]
            row[f"{key}_value"] = value
            row[f"{key}_value_oriented"] = value_oriented
            row[f"{key}_abs_required"] = bool(spec.get("abs_required", True))
            row[f"{key}_rel_required"] = bool(spec.get("rel_required", True))
            row[f"{key}_abs_thr"] = abs_thr
            row[f"{key}_abs_pass"] = bool(abs_pass)
            row[f"{key}_ctrl_mean"] = ctrl_mean
            row[f"{key}_ctrl_std"] = ctrl_std
            row[f"{key}_ctrl_q10"] = ctrl_q10
            row[f"{key}_ctrl_q25"] = ctrl_q25
            row[f"{key}_ctrl_q75"] = ctrl_q75
            row[f"{key}_ctrl_q90"] = ctrl_q90
            row[f"{key}_ctrl_p95"] = ctrl_p95
            row[f"{key}_rel_control_family"] = thresholds["metrics"].get(name, {}).get("control_family_used")
            row[f"{key}_rel_control_family_mapped"] = thresholds["metrics"].get(name, {}).get("control_family")
            row[f"{key}_margin_vs_ctrl_mean"] = margin_ctrl
            row[f"{key}_rel_thr"] = rel_thr
            row[f"{key}_rel_pass"] = bool(rel_pass)
            row[f"{key}_abs_thr_v2"] = abs_thr_v2
            row[f"{key}_abs_rule_v2"] = abs_rule_v2
            row[f"{key}_abs_pass_v2"] = bool(abs_pass_v2)
            row[f"{key}_rel_thr_v2"] = rel_thr
            row[f"{key}_rel_pass_v2"] = bool(rel_pass)
            if spec.get("abs_required", True):
                abs_passes.append(bool(abs_pass))
                abs_passes_v2.append(bool(abs_pass_v2))
                if not bool(abs_pass_v2):
                    fail_flags_v2.append(f"{key}_abs_pass_v2")
            if spec.get("rel_required", True):
                rel_passes.append(bool(rel_pass))
                rel_passes_v2.append(bool(rel_pass))
                if not bool(rel_pass):
                    fail_flags_v2.append(f"{key}_rel_pass_v2")

        # multi-window robustness: require 200-window pass and at least one flank window pass.
        run_dir = row.get("_run_dir")
        window_checks = load_window_checks(run_dir, to_float(row.get("switch_window")) if run_dir else 200)
        window_flags = []
        window_flags_v2 = []
        for w in (100, 200, 400):
            cj = window_checks.get(int(w))
            if not cj:
                row[f"window_{w}_abs_pass"] = False
                row[f"window_{w}_core_abs_pass_v2"] = False
                row[f"window_{w}_fail_reasons_v2"] = "missing_window_checks"
                continue
            temp = {
                "directional_align_overall": cj.get("directional_align_overall"),
                "switch_band_correct_rate": cj.get("switch_band_correct_rate"),
                "switch_margin_pre_signed": to_float(cj.get("switch_margin_pre_signed", cj.get("switch_margin_pre"))),
                "switch_margin_post_signed": to_float(cj.get("switch_margin_post_signed", cj.get("switch_margin_post"))),
                "retained_gap_switch_signed": to_float(cj.get("retained_gap_switch_signed", cj.get("retained_gap_switch"))),
                "peak_delay_min": min(finite_list([cj.get("peak_delay_lambda"), cj.get("peak_delay_gate"), cj.get("peak_delay_rel"), cj.get("peak_delay_min")])) if finite_list([cj.get("peak_delay_lambda"), cj.get("peak_delay_gate"), cj.get("peak_delay_rel"), cj.get("peak_delay_min")]) else np.nan,
                "switch_window": w,
            }
            if (not np.isnan(to_float(temp.get("switch_margin_pre_signed")))) and (not np.isnan(to_float(temp.get("switch_margin_post_signed")))):
                temp["switch_margin_gap_signed"] = float(min(to_float(temp.get("switch_margin_pre_signed")), to_float(temp.get("switch_margin_post_signed"))))
            else:
                temp["switch_margin_gap_signed"] = np.nan
            one_abs = []
            one_abs_v2 = []
            fail_reasons_v2 = []
            for spec in METRIC_SPECS:
                if not spec.get("window_vote", False):
                    continue
                val_w = to_float(temp.get(spec["name"]))
                thr_w = abs_threshold_for(spec, temp)
                thr_w_v2, _ = abs_threshold_for_v2(spec, temp)
                one_abs.append(abs_pass_for(spec, val_w, thr_w))
                pass_w_v2 = abs_pass_for(spec, val_w, thr_w_v2)
                one_abs_v2.append(pass_w_v2)
                if not pass_w_v2:
                    fail_reasons_v2.append(spec["name"])
            pass_w = bool(all(one_abs))
            pass_w_v2 = bool(all(one_abs_v2))
            row[f"window_{w}_abs_pass"] = pass_w
            row[f"window_{w}_core_abs_pass_v2"] = pass_w_v2
            row[f"window_{w}_fail_reasons_v2"] = semicolon_join(fail_reasons_v2)
            window_flags.append(pass_w)
            window_flags_v2.append(pass_w_v2)
        row["window_pass_count"] = int(sum(1 for x in window_flags if x))
        row["window_total"] = 3
        w100 = bool(row.get("window_100_abs_pass"))
        w200 = bool(row.get("window_200_abs_pass"))
        w400 = bool(row.get("window_400_abs_pass"))
        row["window_robust_pass"] = bool(w200 and (w100 or w400))
        row["window_pass_count_v2"] = int(sum(1 for x in window_flags_v2 if x))
        row["window_total_v2"] = 3
        w100_v2 = bool(row.get("window_100_core_abs_pass_v2"))
        w200_v2 = bool(row.get("window_200_core_abs_pass_v2"))
        w400_v2 = bool(row.get("window_400_core_abs_pass_v2"))
        row["window_robust_pass_v2"] = bool(w200_v2 and (w100_v2 or w400_v2))
        if not w200_v2:
            fail_flags_v2.append("window_200_core_abs_pass_v2")
        if not row["window_robust_pass_v2"]:
            if not w100_v2 and not w400_v2:
                fail_flags_v2.append("window_flank_core_abs_pass_v2")
            fail_flags_v2.append("window_robust_pass_v2")

        legacy3 = bool(row.get("gate_direction")) and bool(row.get("high_closer_A0")) and bool(row.get("low_closer_A1"))
        row["legacy3checks"] = legacy3
        row["abs_pass_all"] = bool(all(abs_passes))
        row["rel_pass_all"] = bool(all(rel_passes))
        row["pass_core_checks_v3_abs"] = bool(legacy3 and row["abs_pass_all"])
        row["pass_core_checks_v3"] = bool(legacy3 and row["abs_pass_all"] and row["rel_pass_all"] and row["window_robust_pass"])
        row["abs_pass_all_v2"] = bool(all(abs_passes_v2))
        row["rel_pass_all_v2"] = bool(all(rel_passes_v2))
        row["pass_core_checks_v3_abs_v2"] = bool(legacy3 and row["abs_pass_all_v2"])
        row["pass_core_checks_v3_v2_before_guardrail"] = bool(
            legacy3 and
            row["abs_pass_all_v2"] and
            row["rel_pass_all_v2"] and
            row["window_robust_pass_v2"]
        )
        row["pass_core_checks_v3_v2"] = bool(row["pass_core_checks_v3_v2_before_guardrail"])
        row["fail_reasons_v2"] = semicolon_join(fail_flags_v2)
        row["top_fail_reason_v2"] = fail_flags_v2[0] if fail_flags_v2 else ""

    # Hard guardrail: if too many negative controls pass v3, fail v3 globally.
    neg_rows_eval = [r for r in config_rows if str(r.get("run_type", "")).lower() == "negative_control"]
    neg_v3_pass_count = int(sum(1 for r in neg_rows_eval if bool(r.get("pass_core_checks_v3"))))
    neg_v3_max_allowed = 1
    neg_v3_guardrail_ok = bool(neg_v3_pass_count <= neg_v3_max_allowed)
    for row in config_rows:
        row["pass_core_checks_v3_before_guardrail"] = bool(row.get("pass_core_checks_v3"))
        row["negative_control_v3_pass_count"] = neg_v3_pass_count
        row["negative_control_v3_pass_max_allowed"] = neg_v3_max_allowed
        row["negative_control_v3_guardrail_pass"] = neg_v3_guardrail_ok
        if not neg_v3_guardrail_ok:
            row["pass_core_checks_v3"] = False
            row["v3_guardrail_reason"] = "negative_control_pass_count_exceeded"
        else:
            row["v3_guardrail_reason"] = ""

    neg_v3_v2_pass_count = int(sum(1 for r in neg_rows_eval if bool(r.get("pass_core_checks_v3_v2"))))
    neg_v3_v2_max_allowed = 1
    neg_v3_v2_guardrail_ok = bool(neg_v3_v2_pass_count <= neg_v3_v2_max_allowed)
    for row in config_rows:
        row["pass_core_checks_v3_v2_before_global_guardrail"] = bool(row.get("pass_core_checks_v3_v2"))
        row["negative_control_v3_v2_pass_count"] = neg_v3_v2_pass_count
        row["negative_control_v3_v2_pass_max_allowed"] = neg_v3_v2_max_allowed
        row["negative_control_v3_v2_guardrail_pass"] = neg_v3_v2_guardrail_ok
        if not neg_v3_v2_guardrail_ok:
            row["pass_core_checks_v3_v2"] = False
            row["v3_v2_guardrail_reason"] = "negative_control_pass_count_exceeded"
        else:
            row["v3_v2_guardrail_reason"] = ""

    # Mirror v3-dual fields back to check rows for unified reporting.
    cfg_by_name = {str(r.get("config_name")): r for r in config_rows}
    for row in check_rows:
        src = cfg_by_name.get(str(row.get("config_name")))
        if not src:
            continue
        for k, v in src.items():
            if k.startswith("_"):
                continue
            row[k] = v
    for row in config_rows:
        row.pop("_run_dir", None)
    for row in check_rows:
        row.pop("_run_dir", None)

    cfg_csv = os.path.join(compare_dir, "compare_phaseA_configs.csv")
    cfg_md = os.path.join(compare_dir, "compare_phaseA_configs.md")
    subsets_csv = os.path.join(compare_dir, "compare_phaseA_subsets.csv")
    checks_csv = os.path.join(compare_dir, "compare_phaseA_checks.csv")

    headers_cfg = list(config_rows[0].keys())
    config_rows_norm = [{k: normalize_cell(v) for k, v in r.items()} for r in config_rows]
    with open(cfg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers_cfg)
        w.writeheader()
        for r in config_rows_norm:
            w.writerow(r)

    with open(cfg_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers_cfg) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers_cfg)) + " |\n")
        for r in config_rows_norm:
            f.write("| " + " | ".join([str(r.get(k, "")) for k in headers_cfg]) + " |\n")

    headers_sub = list(subset_rows_out[0].keys()) if subset_rows_out else [
        "lambda_strategy", "run_type", "control_family", "subset", "count", "mean_lambda", "mean_gate_weight",
        "mean_dist_base", "mean_dist_reg0", "mean_dist_reg1", "mean_retained_ratio"
    ]
    subset_rows_norm = [{k: normalize_cell(v) for k, v in r.items()} for r in subset_rows_out]
    with open(subsets_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers_sub)
        w.writeheader()
        for r in subset_rows_norm:
            w.writerow(r)

    headers_check = list(check_rows[0].keys())
    check_rows_norm = [{k: normalize_cell(v) for k, v in r.items()} for r in check_rows]
    with open(checks_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers_check)
        w.writeheader()
        for r in check_rows_norm:
            w.writerow(r)

    block_csv = os.path.join(compare_dir, "compare_phaseA_blockshuffle.csv")
    block_md = os.path.join(compare_dir, "compare_phaseA_blockshuffle.md")
    main_vs_csv = os.path.join(compare_dir, "compare_phaseA_main_vs_blockshuffle.csv")

    block_rows = [r for r in config_rows if str(r.get("run_type", "")).lower() == "negative_control"]
    block_rows_norm = [{k: normalize_cell(v) for k, v in r.items()} for r in block_rows]
    block_headers = list(block_rows[0].keys()) if block_rows else headers_cfg
    with open(block_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=block_headers)
        w.writeheader()
        for r in block_rows_norm:
            w.writerow(r)
    with open(block_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(block_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(block_headers)) + " |\n")
        for r in block_rows_norm:
            f.write("| " + " | ".join([str(r.get(k, "")) for k in block_headers]) + " |\n")

    def mean_metric(rows, key):
        vals = [to_float(r.get(key)) for r in rows]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    def row_peak_delay(r):
        vals = [
            to_float(r.get("peak_delay_lambda")),
            to_float(r.get("peak_delay_gate")),
            to_float(r.get("peak_delay_rel")),
        ]
        vals = [v for v in vals if not np.isnan(v)]
        return float(min(vals)) if vals else np.nan

    block_align_mean = mean_metric(block_rows, "directional_align_overall")
    block_switch_mean = mean_metric(block_rows, "switch_band_correct_rate")
    block_peak_mean = float(np.mean([row_peak_delay(r) for r in block_rows if not np.isnan(row_peak_delay(r))])) if block_rows else np.nan
    main_rows = [r for r in config_rows if str(r.get("run_type", "")).lower() == "main"]
    main_vs_rows = []
    for r in main_rows:
        peak_main = row_peak_delay(r)
        align_main = to_float(r.get("directional_align_overall"))
        switch_main = to_float(r.get("switch_band_correct_rate"))
        out = {
            "config_name": r.get("config_name"),
            "lambda_strategy": r.get("lambda_strategy"),
            "run_type": r.get("run_type"),
            "control_family": r.get("control_family"),
            "switch_window": r.get("switch_window"),
            "directional_align_overall": align_main,
            "switch_band_correct_rate": switch_main,
            "switch_margin_pre_signed": r.get("switch_margin_pre_signed"),
            "switch_margin_post_signed": r.get("switch_margin_post_signed"),
            "switch_margin_pre_abs": r.get("switch_margin_pre_abs"),
            "switch_margin_post_abs": r.get("switch_margin_post_abs"),
            "retained_gap_switch_signed": r.get("retained_gap_switch_signed"),
            "retained_gap_switch_abs": r.get("retained_gap_switch_abs"),
            "peak_delay_main": peak_main,
            "blockshuffle_align_mean": block_align_mean,
            "blockshuffle_switch_mean": block_switch_mean,
            "blockshuffle_peak_delay_mean": block_peak_mean,
            "delta_align_vs_blockshuffle": align_main - block_align_mean if (not np.isnan(align_main) and not np.isnan(block_align_mean)) else np.nan,
            "delta_switch_vs_blockshuffle": switch_main - block_switch_mean if (not np.isnan(switch_main) and not np.isnan(block_switch_mean)) else np.nan,
            "delta_peakdelay_vs_blockshuffle": block_peak_mean - peak_main if (not np.isnan(peak_main) and not np.isnan(block_peak_mean)) else np.nan,
            "abs_pass_all": r.get("abs_pass_all"),
            "rel_pass_all": r.get("rel_pass_all"),
            "window_pass_count": r.get("window_pass_count"),
            "window_total": r.get("window_total"),
            "window_robust_pass": r.get("window_robust_pass"),
            "abs_pass_all_v2": r.get("abs_pass_all_v2"),
            "rel_pass_all_v2": r.get("rel_pass_all_v2"),
            "window_pass_count_v2": r.get("window_pass_count_v2"),
            "window_total_v2": r.get("window_total_v2"),
            "window_robust_pass_v2": r.get("window_robust_pass_v2"),
            "top_fail_reason_v2": r.get("top_fail_reason_v2"),
            "fail_reasons_v2": r.get("fail_reasons_v2"),
            "pass_core_checks_v2": bool(r.get("pass_core_checks_v2")),
            "pass_core_checks_v3_before_guardrail": bool(r.get("pass_core_checks_v3_before_guardrail")),
            "pass_core_checks_v3": bool(r.get("pass_core_checks_v3")),
            "pass_core_checks_v3_v2_before_guardrail": bool(r.get("pass_core_checks_v3_v2_before_guardrail")),
            "pass_core_checks_v3_v2": bool(r.get("pass_core_checks_v3_v2")),
            "negative_control_v3_pass_count": r.get("negative_control_v3_pass_count"),
            "negative_control_v3_pass_max_allowed": r.get("negative_control_v3_pass_max_allowed"),
            "negative_control_v3_guardrail_pass": r.get("negative_control_v3_guardrail_pass"),
            "negative_control_v3_v2_pass_count": r.get("negative_control_v3_v2_pass_count"),
            "negative_control_v3_v2_pass_max_allowed": r.get("negative_control_v3_v2_pass_max_allowed"),
            "negative_control_v3_v2_guardrail_pass": r.get("negative_control_v3_v2_guardrail_pass"),
        }
        for spec in METRIC_SPECS:
            key = spec["name"]
            out[f"{key}_better"] = r.get(f"{key}_better", spec["better"])
            out[f"{key}_value"] = r.get(f"{key}_value", r.get(key))
            out[f"{key}_abs_thr"] = r.get(f"{key}_abs_thr")
            out[f"{key}_abs_pass"] = r.get(f"{key}_abs_pass")
            out[f"{key}_abs_thr_v2"] = r.get(f"{key}_abs_thr_v2")
            out[f"{key}_abs_pass_v2"] = r.get(f"{key}_abs_pass_v2")
            out[f"{key}_rel_control_family"] = r.get(f"{key}_rel_control_family")
            out[f"{key}_ctrl_mean"] = r.get(f"{key}_ctrl_mean")
            out[f"{key}_ctrl_std"] = r.get(f"{key}_ctrl_std")
            out[f"{key}_ctrl_q10"] = r.get(f"{key}_ctrl_q10")
            out[f"{key}_ctrl_q25"] = r.get(f"{key}_ctrl_q25")
            out[f"{key}_ctrl_q75"] = r.get(f"{key}_ctrl_q75")
            out[f"{key}_ctrl_q90"] = r.get(f"{key}_ctrl_q90")
            out[f"{key}_rel_thr"] = r.get(f"{key}_rel_thr")
            out[f"{key}_margin_vs_ctrl_mean"] = r.get(f"{key}_margin_vs_ctrl_mean")
            out[f"{key}_rel_pass"] = r.get(f"{key}_rel_pass")
            out[f"{key}_rel_thr_v2"] = r.get(f"{key}_rel_thr_v2")
            out[f"{key}_rel_pass_v2"] = r.get(f"{key}_rel_pass_v2")
        main_vs_rows.append(out)
    main_vs_headers = list(main_vs_rows[0].keys()) if main_vs_rows else [
        "config_name", "lambda_strategy", "run_type", "control_family",
        "directional_align_overall", "switch_band_correct_rate", "peak_delay_main",
        "blockshuffle_align_mean", "blockshuffle_switch_mean", "blockshuffle_peak_delay_mean",
        "delta_align_vs_blockshuffle", "delta_switch_vs_blockshuffle", "delta_peakdelay_vs_blockshuffle",
        "pass_core_checks_v2", "pass_core_checks_v3"
    ]
    main_vs_norm = [{k: normalize_cell(v) for k, v in r.items()} for r in main_vs_rows]
    with open(main_vs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=main_vs_headers)
        w.writeheader()
        for r in main_vs_norm:
            w.writerow(r)

    # mirror to exports root for convenience
    for src in [cfg_csv, cfg_md, subsets_csv, checks_csv, block_csv, block_md, main_vs_csv, thresholds_json]:
        dst = os.path.join(exports_dir, os.path.basename(src))
        try:
            shutil.copyfile(src, dst)
        except Exception:
            pass

    print(f"[OK] {cfg_csv}")
    print(f"[OK] {cfg_md}")
    print(f"[OK] {subsets_csv}")
    print(f"[OK] {checks_csv}")
    print(f"[OK] {block_csv}")
    print(f"[OK] {block_md}")
    print(f"[OK] {main_vs_csv}")
    print(f"[OK] {thresholds_json}")


if __name__ == "__main__":
    main()
