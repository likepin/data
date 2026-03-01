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
    if not cfg_rows:
        raise RuntimeError("compare_phaseA_configs.csv missing or empty.")
    if not check_rows:
        raise RuntimeError("compare_phaseA_checks.csv missing or empty.")

    main_cfg = [r for r in cfg_rows if str(r.get("run_type", "")).lower() == "main"]
    neg_cfg = [r for r in cfg_rows if str(r.get("run_type", "")).lower() == "negative_control"]
    if not main_cfg:
        main_cfg = cfg_rows

    best_align_row, best_align = argmax_row(main_cfg, "align_overall")
    best_by_align = best_align_row.get("lambda_strategy") if best_align_row else None

    main_checks = [r for r in check_rows if str(r.get("run_type", "")).lower() == "main"]
    neg_checks = [r for r in check_rows if str(r.get("run_type", "")).lower() == "negative_control"]
    if not main_checks:
        main_checks = check_rows

    best_gap_row, best_gap = argmax_row(main_checks, "retained_gap")
    best_by_gap = best_gap_row.get("lambda_strategy") if best_gap_row else None

    pass_vals = [1.0 if to_bool(r.get("pass_core_checks")) else 0.0 for r in main_checks]
    main_pass_rate = float(np.mean(pass_vals)) if pass_vals else np.nan

    main_align_vals = [to_float(r.get("align_overall")) for r in main_checks]
    main_align_vals = [v for v in main_align_vals if not np.isnan(v)]
    neg_align_vals = [to_float(r.get("align_overall")) for r in neg_checks]
    neg_align_vals = [v for v in neg_align_vals if not np.isnan(v)]
    if main_align_vals and neg_align_vals:
        negative_control_drop = float(np.mean(main_align_vals) - np.mean(neg_align_vals))
    else:
        negative_control_drop = np.nan

    summary_json = {
        "best_strategy_by_align": best_by_align,
        "best_strategy_by_retained_gap": best_by_gap,
        "main_runs_pass_rate": main_pass_rate,
        "negative_control_drop": negative_control_drop,
    }

    out_json = os.path.join(exports_dir, "phaseA_summary.json")
    out_md = os.path.join(exports_dir, "phaseA_summary.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # Build a concise markdown summary for report.
    gate_ok = all(to_bool(r.get("gate_direction")) for r in main_checks) if main_checks else False
    high_ok = all(to_bool(r.get("high_closer_A0")) for r in main_checks) if main_checks else False
    low_ok = all(to_bool(r.get("low_closer_A1")) for r in main_checks) if main_checks else False

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("## Phase A Summary\n\n")
        f.write(f"- 最佳门控策略（按 align_overall）: `{best_by_align}`\n")
        f.write(f"- 最佳门控策略（按 retained_gap）: `{best_by_gap}`\n")
        f.write(f"- 主策略 core checks 通过率: `{main_pass_rate:.3f}`\n")
        if not np.isnan(negative_control_drop):
            f.write(f"- 负对照 align_overall 平均下降: `{negative_control_drop:.6f}`\n")
        else:
            f.write("- 负对照 align_overall 平均下降: `nan`\n")
        f.write("\n")
        f.write("### 核心检查\n")
        f.write(f"- gate_direction: {'PASS' if gate_ok else 'FAIL'}\n")
        f.write(f"- high_closer_A0: {'PASS' if high_ok else 'FAIL'}\n")
        f.write(f"- low_closer_A1: {'PASS' if low_ok else 'FAIL'}\n")
        f.write("\n")
        f.write("### 说明\n")
        f.write("- 若主策略显著优于 shuffle/constant，说明 λ 携带结构相关时序信息。\n")

    print(f"[OK] {out_json}")
    print(f"[OK] {out_md}")


if __name__ == "__main__":
    main()
