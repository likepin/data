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
    main_pass_rate_v1 = float(np.mean(pass_v1_vals)) if pass_v1_vals else np.nan
    main_pass_rate_v2 = float(np.mean(pass_v2_vals)) if pass_v2_vals else np.nan
    main_pass_rate_v3 = float(np.mean(pass_v3_vals)) if pass_v3_vals else np.nan

    main_directional_mean = mean_valid(main_checks, "directional_align_overall")
    neg_directional_mean = mean_valid(neg_checks, "directional_align_overall")
    if np.isfinite(main_directional_mean) and np.isfinite(neg_directional_mean):
        negative_control_drop_v2 = float(main_directional_mean - neg_directional_mean)
    else:
        negative_control_drop_v2 = np.nan

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
        "negative_control_drop_v2": negative_control_drop_v2,
        "main_directional_align_mean": main_directional_mean,
        "neg_directional_align_mean": neg_directional_mean,
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
        if not np.isnan(negative_control_drop_v2):
            f.write(f"- Negative-control drop (directional_align_overall): `{negative_control_drop_v2:.6f}`\n")
        else:
            f.write("- Negative-control drop (directional_align_overall): `nan`\n")
        f.write("\n")
        f.write("### V2 Check Summary\n")
        f.write(f"- directional_align_pass: {'PASS' if directional_pass else 'FAIL'}\n")
        f.write(f"- switch_band_pass: {'PASS' if switch_band_pass else 'FAIL'}\n")
        f.write(f"- pass_core_checks_v2: {'PASS' if v2_pass_all else 'FAIL'}\n")
        f.write("\n")
        f.write("### Notes\n")
        f.write("- v2 ranking uses switch-aware metrics to better separate true temporal alignment from shuffle/constant/shift controls.\n")

    out_block_md = os.path.join(exports_dir, "phaseA_blockshuffle_summary.md")
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
            f.write("| config_name | lambda_strategy | delta_align_vs_blockshuffle | delta_switch_vs_blockshuffle | delta_peakdelay_vs_blockshuffle | pass_core_checks_v3 |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for r in main_vs_rows:
                f.write(
                    f"| {r.get('config_name')} | {r.get('lambda_strategy')} | {r.get('delta_align_vs_blockshuffle')} | "
                    f"{r.get('delta_switch_vs_blockshuffle')} | {r.get('delta_peakdelay_vs_blockshuffle')} | {r.get('pass_core_checks_v3')} |\n"
                )
    else:
        with open(out_block_md, "w", encoding="utf-8") as f:
            f.write("## Phase A Block-Shuffle Summary\n\n")
            f.write("- Missing compare_phaseA_main_vs_blockshuffle.csv or compare_phaseA_blockshuffle.csv.\n")

    print(f"[OK] {out_json}")
    print(f"[OK] {out_md}")
    print(f"[OK] {out_block_md}")


if __name__ == "__main__":
    main()
