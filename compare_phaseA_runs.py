import os
import csv
import json
import argparse
import shutil

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
        switch_band_pass = to_bool(pick_metric(sanity, checks, "switch_band_pass"))
        directional_align_pass = to_bool(pick_metric(sanity, checks, "directional_align_pass"))
        switch_margin_pass = to_bool(pick_metric(sanity, checks, "switch_margin_pass"))
        peak_delay_pass = to_bool(pick_metric(sanity, checks, "peak_delay_pass"))
        retained_gap_switch_pass = to_bool(pick_metric(sanity, checks, "retained_gap_switch_pass"))

        config_rows.append(
            {
                "config_name": run_name,
                "lambda_strategy": lambda_strategy,
                "run_type": run_type,
                "control_family": control_family,
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
                "switch_band_pass": bool(switch_band_pass),
                "directional_align_pass": bool(directional_align_pass),
                "switch_margin_pass": bool(switch_margin_pass),
                "peak_delay_pass": bool(peak_delay_pass),
                "retained_gap_switch_pass": bool(retained_gap_switch_pass),
                "pass_core_checks_v2": pass_core_v2,
                "pass_core_checks_v3": pass_core_v3,
            }
        )

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

        check_rows.append(
            {
                "lambda_strategy": lambda_strategy,
                "run_type": run_type,
                "control_family": control_family,
                "gate_direction": bool(gate_direction),
                "high_closer_A0": bool(high_closer),
                "low_closer_A1": bool(low_closer),
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
                "switch_band_pass": bool(switch_band_pass),
                "directional_align_pass": bool(directional_align_pass),
                "switch_margin_pass": bool(switch_margin_pass),
                "peak_delay_pass": bool(peak_delay_pass),
                "retained_gap_switch_pass": bool(retained_gap_switch_pass),
                "pass_core_checks": pass_core,
                "pass_core_checks_v2": pass_core_v2,
                "pass_core_checks_v3": pass_core_v3,
            }
        )

    if not config_rows:
        raise RuntimeError("No valid run results found in runs_dir.")

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
        main_vs_rows.append(
            {
                "config_name": r.get("config_name"),
                "lambda_strategy": r.get("lambda_strategy"),
                "run_type": r.get("run_type"),
                "control_family": r.get("control_family"),
                "directional_align_overall": align_main,
                "switch_band_correct_rate": switch_main,
                "peak_delay_main": peak_main,
                "blockshuffle_align_mean": block_align_mean,
                "blockshuffle_switch_mean": block_switch_mean,
                "blockshuffle_peak_delay_mean": block_peak_mean,
                "delta_align_vs_blockshuffle": align_main - block_align_mean if (not np.isnan(align_main) and not np.isnan(block_align_mean)) else np.nan,
                "delta_switch_vs_blockshuffle": switch_main - block_switch_mean if (not np.isnan(switch_main) and not np.isnan(block_switch_mean)) else np.nan,
                "delta_peakdelay_vs_blockshuffle": block_peak_mean - peak_main if (not np.isnan(peak_main) and not np.isnan(block_peak_mean)) else np.nan,
                "pass_core_checks_v2": bool(r.get("pass_core_checks_v2")),
                "pass_core_checks_v3": bool(r.get("pass_core_checks_v3")),
            }
        )
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
    for src in [cfg_csv, cfg_md, subsets_csv, checks_csv, block_csv, block_md, main_vs_csv]:
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


if __name__ == "__main__":
    main()
