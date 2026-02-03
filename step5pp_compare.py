import os
import json
import argparse
import subprocess
import sys
import csv


def run_sim(data_dir, cfg_path, out_dir):
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
    subprocess.check_call(cmd)


def read_summary(summary_path):
    rows = []
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def pick_row(rows, subset_name):
    for r in rows:
        if r.get("subset") == subset_name:
            return r
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--cfg_a", type=str, required=True)
    parser.add_argument("--cfg_b", type=str, required=True)
    parser.add_argument("--cfg_c", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "exports_step5pp")
    os.makedirs(out_dir, exist_ok=True)

    out_a = os.path.join(out_dir, "compare_a")
    out_b = os.path.join(out_dir, "compare_b")
    out_c = os.path.join(out_dir, "compare_c")

    run_sim(args.data_dir, args.cfg_a, out_a)
    run_sim(args.data_dir, args.cfg_b, out_b)
    if args.cfg_c:
        run_sim(args.data_dir, args.cfg_c, out_c)

    cfg_a = json.load(open(os.path.join(out_a, "config_used.json"), "r", encoding="utf-8"))
    cfg_b = json.load(open(os.path.join(out_b, "config_used.json"), "r", encoding="utf-8"))

    sum_a = read_summary(os.path.join(out_a, "step5pp_summary.csv"))
    sum_b = read_summary(os.path.join(out_b, "step5pp_summary.csv"))

    a_high = pick_row(sum_a, "high_non_sat")
    a_low = pick_row(sum_a, "low")
    b_high = pick_row(sum_b, "high_non_sat")
    b_low = pick_row(sum_b, "low")

    low_post_min = int(cfg_a.get("low_post_min", 10))
    rows = [
        {
            "name": "cfg_a",
            "edge_mask": cfg_a.get("edge_mask"),
            "delta_mask_mode": cfg_a.get("delta_mask_mode"),
            "dist_mask_mode": cfg_a.get("dist_mask_mode"),
            "dist_std_base": cfg_a.get("dist_std_base"),
            "dist_std_reg0": cfg_a.get("dist_std_reg0"),
            "dist_std_reg1": cfg_a.get("dist_std_reg1"),
            "mean_dist_base_high": to_float(a_high.get("mean_dist_base")) if a_high else 0.0,
            "mean_dist_base_low": to_float(a_low.get("mean_dist_base")) if a_low else 0.0,
            "mean_retained_high": to_float(a_high.get("mean_retained_ratio")) if a_high else 0.0,
            "mean_retained_low": to_float(a_low.get("mean_retained_ratio")) if a_low else 0.0,
            "align_low_post": cfg_a.get("align_low_post"),
            "n_low_post": cfg_a.get("n_low_post"),
            "dist_mask_nnz": cfg_a.get("dist_mask_nnz"),
        },
        {
            "name": "cfg_b",
            "edge_mask": cfg_b.get("edge_mask"),
            "delta_mask_mode": cfg_b.get("delta_mask_mode"),
            "dist_mask_mode": cfg_b.get("dist_mask_mode"),
            "dist_std_base": cfg_b.get("dist_std_base"),
            "dist_std_reg0": cfg_b.get("dist_std_reg0"),
            "dist_std_reg1": cfg_b.get("dist_std_reg1"),
            "mean_dist_base_high": to_float(b_high.get("mean_dist_base")) if b_high else 0.0,
            "mean_dist_base_low": to_float(b_low.get("mean_dist_base")) if b_low else 0.0,
            "mean_retained_high": to_float(b_high.get("mean_retained_ratio")) if b_high else 0.0,
            "mean_retained_low": to_float(b_low.get("mean_retained_ratio")) if b_low else 0.0,
            "align_low_post": cfg_b.get("align_low_post"),
            "n_low_post": cfg_b.get("n_low_post"),
            "dist_mask_nnz": cfg_b.get("dist_mask_nnz"),
        },
    ]
    if args.cfg_c is None:
        default_c = os.path.join(args.data_dir, "exports_step5pp", "cfg_union_delta_topk.json")
        if os.path.isfile(default_c):
            args.cfg_c = default_c

    if args.cfg_c:
        cfg_c = json.load(open(os.path.join(out_c, "config_used.json"), "r", encoding="utf-8"))
        sum_c = read_summary(os.path.join(out_c, "step5pp_summary.csv"))
        c_high = pick_row(sum_c, "high_non_sat")
        c_low = pick_row(sum_c, "low")
        rows.append({
            "name": "cfg_c",
            "edge_mask": cfg_c.get("edge_mask"),
            "delta_mask_mode": cfg_c.get("delta_mask_mode"),
            "dist_mask_mode": cfg_c.get("dist_mask_mode"),
            "dist_std_base": cfg_c.get("dist_std_base"),
            "dist_std_reg0": cfg_c.get("dist_std_reg0"),
            "dist_std_reg1": cfg_c.get("dist_std_reg1"),
            "mean_dist_base_high": to_float(c_high.get("mean_dist_base")) if c_high else 0.0,
            "mean_dist_base_low": to_float(c_low.get("mean_dist_base")) if c_low else 0.0,
            "mean_retained_high": to_float(c_high.get("mean_retained_ratio")) if c_high else 0.0,
            "mean_retained_low": to_float(c_low.get("mean_retained_ratio")) if c_low else 0.0,
            "align_low_post": cfg_c.get("align_low_post"),
            "n_low_post": cfg_c.get("n_low_post"),
            "dist_mask_nnz": cfg_c.get("dist_mask_nnz"),
        })

    out_csv = os.path.join(out_dir, "compare_masks.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        header = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_md = os.path.join(out_dir, "compare_masks.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(rows[0].keys()) + " |\n")
        f.write("| " + " | ".join(["---"] * len(rows[0])) + " |\n")
        for r in rows:
            f.write("| " + " | ".join([str(r[k]) for k in rows[0].keys()]) + " |\n")

    # quick conclusions
    try:
        inc = float(rows[1]["dist_std_reg0"]) > float(rows[0]["dist_std_reg0"])
    except Exception:
        inc = False
    try:
        low_post_ok = float(rows[0].get("n_low_post", 0)) >= low_post_min
    except Exception:
        low_post_ok = False
    try:
        meaningful = float(rows[1].get("align_low_post", 0)) == float(rows[1].get("align_low_post", 0))
    except Exception:
        meaningful = False

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")
    print(f"union increases dist_std_reg? = {inc}")
    print(f"low_post_count sufficient? = {low_post_ok}")
    print(f"align_low_post meaningful? = {meaningful}")


if __name__ == "__main__":
    main()
