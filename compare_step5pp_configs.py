import os
import json
import csv
import argparse
import subprocess
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(data_dir, cfg_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "step5pp_simulate_gated_graph.py",
        "--data_dir",
        data_dir,
        "--config",
        cfg_path,
        "--out_dir",
        out_dir,
        "--sanity",
    ]
    subprocess.check_call(cmd)


def resolve_cfg(path, data_dir):
    if os.path.isfile(path):
        return path
    alt = os.path.join(data_dir, path)
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError(f"Config not found: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--cfg_base", type=str, default="cfg_base_only.json")
    parser.add_argument("--cfg_union", type=str, default="cfg_union.json")
    parser.add_argument("--cfg_union_delta_topk", type=str, default="cfg_union_delta_topk.json")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "exports_step5pp")
    os.makedirs(out_dir, exist_ok=True)

    cfg_map = {
        "cfg_base_only": resolve_cfg(args.cfg_base, args.data_dir),
        "cfg_union": resolve_cfg(args.cfg_union, args.data_dir),
        "cfg_union_delta_topk": resolve_cfg(args.cfg_union_delta_topk, args.data_dir),
    }

    rows = []
    for name, cfg_path in cfg_map.items():
        run_dir = os.path.join(out_dir, f"compare_{name}")
        run_one(args.data_dir, cfg_path, run_dir)
        diag = load_json(os.path.join(run_dir, "step5pp_diagnostics.json"))
        used = load_json(os.path.join(run_dir, "config_used.json"))
        rows.append(
            {
                "config_name": name,
                "delta_mask_mode": used.get("delta_mask_mode", diag.get("delta_mask_mode")),
                "dist_mask_mode": used.get("dist_mask_mode", diag.get("dist_mask_mode")),
                "delta_mask_nnz": diag.get("delta_mask_nnz"),
                "dist_mask_nnz": diag.get("dist_mask_nnz"),
                "dist_std_base": diag.get("dist_std_base"),
                "dist_std_reg0": diag.get("dist_std_reg0"),
                "dist_std_reg1": diag.get("dist_std_reg1"),
                "align_pre": diag.get("align_pre", diag.get("align_all_pre")),
                "align_post": diag.get("align_post", diag.get("align_all_post")),
                "align_overall": diag.get("align_overall", diag.get("overall_align")),
                "high_mean_lambda": diag.get("mean_lambda_high"),
                "high_mean_gate_weight": diag.get("mean_gate_high"),
                "low_mean_lambda": diag.get("mean_lambda_low"),
                "low_mean_gate_weight": diag.get("mean_gate_low"),
                "high_closer_A0": diag.get("check_high_closer_a0"),
                "low_closer_A1": diag.get("check_low_closer_a1"),
                "regime_swapped": used.get("regime_swapped"),
                "swap_reason": used.get("swap_reason"),
            }
        )

    header = list(rows[0].keys())
    out_csv = os.path.join(out_dir, "compare_configs.csv")
    out_md = os.path.join(out_dir, "compare_configs.md")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in header) + " |\n")

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


if __name__ == "__main__":
    main()
