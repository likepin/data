import os
import json
import csv
import argparse
import subprocess
import sys


def safe_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)


def run_one(data_dir, cfg_path, out_dir, sanity):
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
    ]
    if sanity:
        cmd.append("--sanity")
    subprocess.check_call(cmd)


def load_json_or_empty(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("configs", nargs="+", help="List of config JSON paths.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "exports_step5pp", "compare_configs")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for i, cfg_path in enumerate(args.configs, start=1):
        run_name = f"{i:02d}_{safe_name(cfg_path)}"
        run_dir = os.path.join(out_dir, run_name)
        run_one(args.data_dir, cfg_path, run_dir, sanity=args.sanity)

        diagnostics = load_json_or_empty(os.path.join(run_dir, "step5pp_diagnostics.json"))
        config_used = load_json_or_empty(os.path.join(run_dir, "config_used.json"))

        row = {
            "run_name": run_name,
            "config_path": cfg_path,
            "delta_mask_mode": config_used.get("delta_mask_mode", diagnostics.get("delta_mask_mode")),
            "dist_mask_mode": config_used.get("dist_mask_mode", diagnostics.get("dist_mask_mode")),
            "regime_ref_source": config_used.get("regime_ref_source"),
            "regime_swapped": config_used.get("regime_swapped"),
            "swap_reason": config_used.get("swap_reason"),
            "overall_align": diagnostics.get("overall_align", config_used.get("overall_align")),
            "align_all_pre": diagnostics.get("align_all_pre", config_used.get("align_all_pre")),
            "align_all_post": diagnostics.get("align_all_post", config_used.get("align_all_post")),
            "mean_margin_pre": diagnostics.get("mean_margin_pre", config_used.get("mean_margin_pre")),
            "mean_margin_post": diagnostics.get("mean_margin_post", config_used.get("mean_margin_post")),
            "rel_pre_mean": diagnostics.get("rel_pre_mean", config_used.get("rel_pre_mean")),
            "rel_post_mean": diagnostics.get("rel_post_mean", config_used.get("rel_post_mean")),
            "delta_mask_nnz": diagnostics.get("delta_mask_nnz", config_used.get("delta_mask_nnz")),
            "dist_mask_nnz": diagnostics.get("dist_mask_nnz", config_used.get("dist_mask_nnz")),
            "check_gate_direction": diagnostics.get("check_gate_direction", config_used.get("check_gate_direction")),
            "check_high_closer_a0": diagnostics.get("check_high_closer_a0", config_used.get("check_high_closer_a0")),
            "check_low_closer_a1": diagnostics.get("check_low_closer_a1", config_used.get("check_low_closer_a1")),
            "check_pre_post_direction": diagnostics.get("check_pre_post_direction", config_used.get("check_pre_post_direction")),
            "check_overall_pass": diagnostics.get("check_overall_pass", config_used.get("check_overall_pass")),
            "n_low_post": diagnostics.get("n_low_post", config_used.get("n_low_post")),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No configs were processed.")

    out_csv = os.path.join(out_dir, "compare_configs.csv")
    out_md = os.path.join(out_dir, "compare_configs.md")
    headers = list(rows[0].keys())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join([str(r[h]) for h in headers]) + " |\n")

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


if __name__ == "__main__":
    main()
