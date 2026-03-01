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
    if "shuffle" in name or "constant" in name or "const" in name:
        return "negative_control"
    return "main"


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

    for run_name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_dir):
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

        lambda_strategy = sanity.get("config_name") or config_used.get("lambda_tag") or run_name
        run_type = config_used.get("run_type") or infer_run_type(run_name)

        gate_direction = to_bool(checks.get("gate_direction", sanity.get("gate_direction")))
        high_closer = to_bool(checks.get("high_closer_A0", sanity.get("high_closer_A0")))
        low_closer = to_bool(checks.get("low_closer_A1", sanity.get("low_closer_A1")))
        pass_core = bool(gate_direction and high_closer and low_closer)

        config_rows.append(
            {
                "config_name": run_name,
                "lambda_strategy": lambda_strategy,
                "run_type": run_type,
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
            }
        )

        for r in subset_rows:
            subset_rows_out.append(
                {
                    "lambda_strategy": lambda_strategy,
                    "run_type": run_type,
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
                "gate_direction": gate_direction,
                "high_closer_A0": high_closer,
                "low_closer_A1": low_closer,
                "align_overall": sanity.get("align_overall", sanity.get("overall_align")),
                "margin_pre": sanity.get("margin_pre", sanity.get("mean_margin_pre")),
                "margin_post": sanity.get("margin_post", sanity.get("mean_margin_post")),
                "retained_gap": retained_gap,
                "pass_core_checks": pass_core,
            }
        )

    if not config_rows:
        raise RuntimeError("No valid run results found in runs_dir.")

    cfg_csv = os.path.join(compare_dir, "compare_phaseA_configs.csv")
    cfg_md = os.path.join(compare_dir, "compare_phaseA_configs.md")
    subsets_csv = os.path.join(compare_dir, "compare_phaseA_subsets.csv")
    checks_csv = os.path.join(compare_dir, "compare_phaseA_checks.csv")

    headers_cfg = list(config_rows[0].keys())
    with open(cfg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers_cfg)
        w.writeheader()
        for r in config_rows:
            w.writerow(r)

    with open(cfg_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers_cfg) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers_cfg)) + " |\n")
        for r in config_rows:
            f.write("| " + " | ".join([str(r.get(k, "")) for k in headers_cfg]) + " |\n")

    headers_sub = list(subset_rows_out[0].keys()) if subset_rows_out else [
        "lambda_strategy", "run_type", "subset", "count", "mean_lambda", "mean_gate_weight",
        "mean_dist_base", "mean_dist_reg0", "mean_dist_reg1", "mean_retained_ratio"
    ]
    with open(subsets_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers_sub)
        w.writeheader()
        for r in subset_rows_out:
            w.writerow(r)

    headers_check = list(check_rows[0].keys())
    with open(checks_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers_check)
        w.writeheader()
        for r in check_rows:
            w.writerow(r)

    # mirror to exports root for convenience
    for src in [cfg_csv, cfg_md, subsets_csv, checks_csv]:
        dst = os.path.join(exports_dir, os.path.basename(src))
        try:
            shutil.copyfile(src, dst)
        except Exception:
            pass

    print(f"[OK] {cfg_csv}")
    print(f"[OK] {cfg_md}")
    print(f"[OK] {subsets_csv}")
    print(f"[OK] {checks_csv}")


if __name__ == "__main__":
    main()
