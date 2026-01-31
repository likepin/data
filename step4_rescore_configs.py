import os
import csv
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append(r)
    return rows


def to_float(v):
    try:
        if v is None or v == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def to_int(v):
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except Exception:
        return None


def robust_minmax(values, lo=0.05, hi=0.95, eps=1e-12):
    arr = np.array(values, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return np.full_like(arr, 0.5)
    q_lo = np.quantile(arr[mask], lo)
    q_hi = np.quantile(arr[mask], hi)
    clipped = arr.copy()
    clipped[mask] = np.clip(arr[mask], q_lo, q_hi)
    vmin = clipped[mask].min()
    vmax = clipped[mask].max()
    if vmax - vmin < eps:
        out = np.full_like(arr, 0.5)
    else:
        out = (clipped - vmin) / (vmax - vmin)
    out[~mask] = np.nan
    return out


def robust_z(values, lo=0.05, hi=0.95, eps=1e-12):
    arr = np.array(values, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return np.full_like(arr, 0.5)
    q_lo = np.quantile(arr[mask], lo)
    q_hi = np.quantile(arr[mask], hi)
    wins = arr.copy()
    wins[mask] = np.clip(arr[mask], q_lo, q_hi)
    mu = wins[mask].mean()
    sd = wins[mask].std()
    if sd < eps:
        z = np.zeros_like(arr)
    else:
        z = (wins - mu) / sd
    # map to 0..1 via minmax on z
    z_mask = np.isfinite(z)
    if not z_mask.any():
        out = np.full_like(arr, 0.5)
    else:
        zmin = z[z_mask].min()
        zmax = z[z_mask].max()
        if zmax - zmin < eps:
            out = np.full_like(arr, 0.5)
        else:
            out = (z - zmin) / (zmax - zmin)
    out[~mask] = np.nan
    return out


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    header = sorted(keys)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def write_md(rows, out_path, title):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = [
        "window", "k", "score_equal", "score_gating", "score_regime",
        "auc_regime", "corr_mse", "sep_mean", "top10_in_reg1",
        "smooth_mean_abs_diff", "smooth_std_diff"
    ]
    lines = [f"## {title}\n",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join([
            str(r.get("window", "")),
            str(r.get("k", "")),
            f"{float(r.get('score_equal', np.nan)):.6f}",
            f"{float(r.get('score_gating', np.nan)):.6f}",
            f"{float(r.get('score_regime', np.nan)):.6f}",
            f"{float(r.get('auc_regime', np.nan)):.6f}",
            f"{float(r.get('corr_mse', np.nan)):.6f}",
            f"{float(r.get('sep_mean', np.nan)):.6f}",
            f"{float(r.get('top10_in_reg1', np.nan)):.6f}",
            f"{float(r.get('smooth_mean_abs_diff', np.nan)):.6f}",
            f"{float(r.get('smooth_std_diff', np.nan)):.6f}",
        ]) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def topk_table(rows, score_key, k=5):
    ordered = sorted(rows, key=lambda r: (r.get(score_key) if np.isfinite(r.get(score_key, np.nan)) else -1e9), reverse=True)
    return ordered[:k]


def plot_component_contrib(row, score_name, weights, out_path):
    comps = [
        ("n_auc", row.get("n_auc")),
        ("n_corr", row.get("n_corr")),
        ("n_sep", row.get("n_sep")),
        ("n_top10", row.get("n_top10")),
        ("1-n_smooth_mean", 1.0 - row.get("n_smooth_mean")),
        ("1-n_smooth_std", 1.0 - row.get("n_smooth_std")),
    ]
    labels = []
    vals = []
    for (name, val) in comps:
        labels.append(name)
        vals.append(float(val))
    w = np.array(weights, dtype=float)
    contrib = w * np.array(vals, dtype=float)

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(len(labels)), contrib, color="tab:blue")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("weighted contribution")
    ax.set_title(f"{score_name} component contributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--norm_mode", type=str, default="robust_minmax", choices=["robust_minmax", "robust_z"])
    parser.add_argument("--norm_lo", type=float, default=0.05)
    parser.add_argument("--norm_hi", type=float, default=0.95)
    parser.add_argument("--no_filters", action="store_true")
    parser.add_argument("--smooth_mean_q", type=float, default=0.60)
    parser.add_argument("--smooth_std_q", type=float, default=0.60)
    parser.add_argument("--smooth_mean_max", type=float, default=None)
    parser.add_argument("--smooth_std_max", type=float, default=None)
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    in_csvs = args.in_csv
    if args.out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(in_csvs[0]))
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for path in in_csvs:
        src = os.path.basename(path)
        for r in read_csv(path):
            r2 = dict(r)
            if len(in_csvs) > 1:
                r2["source"] = src
            rows.append(r2)

    # parse numeric fields
    for r in rows:
        r["window"] = to_int(r.get("window"))
        r["k"] = to_int(r.get("k"))
        r["auc_regime"] = to_float(r.get("auc_regime"))
        r["corr_mse"] = to_float(r.get("corr_mse"))
        r["corr_mae"] = to_float(r.get("corr_mae"))
        r["sep_mean"] = to_float(r.get("sep_mean"))
        r["sep_median"] = to_float(r.get("sep_median"))
        r["top10_in_reg1"] = to_float(r.get("top10_in_reg1"))
        r["reg1_top10_coverage"] = to_float(r.get("reg1_top10_coverage"))
        r["smooth_std_diff"] = to_float(r.get("smooth_std_diff"))
        r["smooth_mean_abs_diff"] = to_float(r.get("smooth_mean_abs_diff"))
        r["valid_ratio"] = to_float(r.get("valid_ratio"))

    # normalization
    norm_fn = robust_minmax if args.norm_mode == "robust_minmax" else robust_z
    n_auc = norm_fn([r["auc_regime"] for r in rows], args.norm_lo, args.norm_hi)
    n_corr = norm_fn([r["corr_mse"] for r in rows], args.norm_lo, args.norm_hi)
    n_sep = norm_fn([r["sep_mean"] for r in rows], args.norm_lo, args.norm_hi)
    n_top10 = norm_fn([r["top10_in_reg1"] for r in rows], args.norm_lo, args.norm_hi)
    n_smooth_mean = norm_fn([r["smooth_mean_abs_diff"] for r in rows], args.norm_lo, args.norm_hi)
    n_smooth_std = norm_fn([r["smooth_std_diff"] for r in rows], args.norm_lo, args.norm_hi)

    for i, r in enumerate(rows):
        r["n_auc"] = float(n_auc[i])
        r["n_corr"] = float(n_corr[i])
        r["n_sep"] = float(n_sep[i])
        r["n_top10"] = float(n_top10[i])
        r["n_smooth_mean"] = float(n_smooth_mean[i])
        r["n_smooth_std"] = float(n_smooth_std[i])

    # filters
    filtered_out = []
    kept = []
    if args.no_filters:
        for r in rows:
            r["passed_filters"] = True
            kept.append(r)
    else:
        smooth_mean_vals = np.array([r["smooth_mean_abs_diff"] for r in rows], dtype=float)
        smooth_std_vals = np.array([r["smooth_std_diff"] for r in rows], dtype=float)
        if args.smooth_mean_max is not None:
            smooth_mean_thr = float(args.smooth_mean_max)
        else:
            smooth_mean_thr = float(np.quantile(smooth_mean_vals[np.isfinite(smooth_mean_vals)], args.smooth_mean_q))
        if args.smooth_std_max is not None:
            smooth_std_thr = float(args.smooth_std_max)
        else:
            smooth_std_thr = float(np.quantile(smooth_std_vals[np.isfinite(smooth_std_vals)], args.smooth_std_q))

        for r in rows:
            reasons = []
            sm = r["smooth_mean_abs_diff"]
            ss = r["smooth_std_diff"]
            cm = r["corr_mse"]
            vr = r.get("valid_ratio", np.nan)

            if not np.isfinite(sm) or sm > smooth_mean_thr:
                reasons.append("smooth_mean_abs_diff")
            if not np.isfinite(ss) or ss > smooth_std_thr:
                reasons.append("smooth_std_diff")
            if not np.isfinite(cm) or cm < 0:
                reasons.append("corr_mse")
            if np.isfinite(vr) and vr < 0.95:
                reasons.append("valid_ratio")

            if reasons:
                r2 = dict(r)
                r2["filter_reasons"] = ";".join(reasons)
                filtered_out.append(r2)
                r["passed_filters"] = False
            else:
                r["passed_filters"] = True
                kept.append(r)

    # scoring
    for r in rows:
        n_auc_v = r["n_auc"]
        n_corr_v = r["n_corr"]
        n_sep_v = r["n_sep"]
        n_top10_v = r["n_top10"]
        n_sm = r["n_smooth_mean"]
        n_ss = r["n_smooth_std"]
        r["score_equal"] = float(
            n_auc_v + (1 - n_sm) + n_top10_v + n_corr_v + n_sep_v + (1 - n_ss)
        )
        r["score_gating"] = float(
            0.55 * n_corr_v +
            0.15 * n_auc_v +
            0.10 * n_top10_v +
            0.10 * n_sep_v +
            0.05 * (1 - n_sm) +
            0.05 * (1 - n_ss)
        )
        r["score_regime"] = float(
            0.40 * n_auc_v +
            0.20 * n_sep_v +
            0.20 * n_corr_v +
            0.10 * n_top10_v +
            0.05 * (1 - n_sm) +
            0.05 * (1 - n_ss)
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rescored_csv = os.path.join(out_dir, f"rescored_results_{ts}.csv")
    rescored_md = os.path.join(out_dir, f"rescored_results_{ts}.md")
    write_csv(kept, rescored_csv)
    write_md(kept, rescored_md, title="Rescored Results")

    if len(in_csvs) > 1:
        merged_path = os.path.join(out_dir, f"merged_rescored_results_{ts}.csv")
        write_csv(kept, merged_path)
    else:
        merged_path = None

    filtered_path = os.path.join(out_dir, f"filtered_out_{ts}.csv")
    if filtered_out:
        write_csv(filtered_out, filtered_path)

    topn = max(1, int(args.top_n))
    top_equal = topk_table(kept, "score_equal", k=topn)
    top_gating = topk_table(kept, "score_gating", k=topn)
    top_regime = topk_table(kept, "score_regime", k=topn)

    top_txt = os.path.join(out_dir, f"rescored_top{topn}_{ts}.txt")
    with open(top_txt, "w", encoding="utf-8") as f:
        f.write("Top configs by score_equal:\n")
        for i, r in enumerate(top_equal, start=1):
            f.write(f"{i:2d}. window={r.get('window')} k={r.get('k')} score_equal={r.get('score_equal'):.6f}\n")
        f.write("\nTop configs by score_gating:\n")
        for i, r in enumerate(top_gating, start=1):
            f.write(f"{i:2d}. window={r.get('window')} k={r.get('k')} score_gating={r.get('score_gating'):.6f}\n")
        f.write("\nTop configs by score_regime:\n")
        for i, r in enumerate(top_regime, start=1):
            f.write(f"{i:2d}. window={r.get('window')} k={r.get('k')} score_regime={r.get('score_regime'):.6f}\n")

    # selection report
    report_path = os.path.join(out_dir, f"selection_report_{ts}.md")
    top5_equal = topk_table(kept, "score_equal", k=5)
    top5_gating = topk_table(kept, "score_gating", k=5)
    top5_regime = topk_table(kept, "score_regime", k=5)

    def keypair(r):
        return (r.get("window"), r.get("k"))

    set_equal = set(keypair(r) for r in top5_equal)
    set_gating = set(keypair(r) for r in top5_gating)
    set_regime = set(keypair(r) for r in top5_regime)
    common_top5 = set_equal & set_gating & set_regime

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Selection Report\n\n")
        f.write("## Top-5 by score_equal\n\n")
        f.write("| rank | window | k | score_equal | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for i, r in enumerate(top5_equal, start=1):
            f.write(f"| {i} | {r.get('window')} | {r.get('k')} | {r.get('score_equal'):.6f} | "
                    f"{r.get('auc_regime'):.6f} | {r.get('corr_mse'):.6f} | {r.get('sep_mean'):.6f} | "
                    f"{r.get('top10_in_reg1'):.6f} | {r.get('smooth_mean_abs_diff'):.6f} | {r.get('smooth_std_diff'):.6f} |\n")

        f.write("\n## Top-5 by score_gating\n\n")
        f.write("| rank | window | k | score_gating | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for i, r in enumerate(top5_gating, start=1):
            f.write(f"| {i} | {r.get('window')} | {r.get('k')} | {r.get('score_gating'):.6f} | "
                    f"{r.get('auc_regime'):.6f} | {r.get('corr_mse'):.6f} | {r.get('sep_mean'):.6f} | "
                    f"{r.get('top10_in_reg1'):.6f} | {r.get('smooth_mean_abs_diff'):.6f} | {r.get('smooth_std_diff'):.6f} |\n")

        f.write("\n## Top-5 by score_regime\n\n")
        f.write("| rank | window | k | score_regime | auc | corr_mse | sep_mean | top10 | smooth_mean | smooth_std |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for i, r in enumerate(top5_regime, start=1):
            f.write(f"| {i} | {r.get('window')} | {r.get('k')} | {r.get('score_regime'):.6f} | "
                    f"{r.get('auc_regime'):.6f} | {r.get('corr_mse'):.6f} | {r.get('sep_mean'):.6f} | "
                    f"{r.get('top10_in_reg1'):.6f} | {r.get('smooth_mean_abs_diff'):.6f} | {r.get('smooth_std_diff'):.6f} |\n")

        f.write("\n## Top-5 consistency\n\n")
        f.write(f"- Common configs across all three top-5: {sorted(list(common_top5))}\n")

    # component contribution plots
    equal_weights = [1, 1, 1, 1, 1, 1]
    gating_weights = [0.15, 0.55, 0.10, 0.10, 0.05, 0.05]
    regime_weights = [0.40, 0.20, 0.20, 0.10, 0.05, 0.05]
    plot_component_contrib(top5_equal[0], "score_equal", equal_weights, os.path.join(out_dir, "contrib_equal.png"))
    plot_component_contrib(top5_gating[0], "score_gating", gating_weights, os.path.join(out_dir, "contrib_gating.png"))
    plot_component_contrib(top5_regime[0], "score_regime", regime_weights, os.path.join(out_dir, "contrib_regime.png"))

    # pareto plots
    smooth = np.array([r.get("smooth_mean_abs_diff") for r in kept], dtype=float)
    aucs = np.array([r.get("auc_regime") for r in kept], dtype=float)
    corrs = np.array([r.get("corr_mse") for r in kept], dtype=float)
    scores = np.array([r.get("score_gating") for r in kept], dtype=float)

    pareto1 = os.path.join(out_dir, "pareto_auc_vs_smooth.png")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(smooth, aucs, c=scores, cmap="viridis", s=35, edgecolors="none")
    ax.set_xlabel("smooth_mean_abs_diff (lower is better)")
    ax.set_ylabel("auc_regime (higher is better)")
    ax.set_title("Pareto: AUC vs Smooth")
    fig.colorbar(sc, ax=ax, shrink=0.85, label="score_gating")
    fig.tight_layout()
    fig.savefig(pareto1, dpi=200)
    plt.close(fig)

    pareto2 = os.path.join(out_dir, "pareto_corr_vs_smooth.png")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(smooth, corrs, c=scores, cmap="viridis", s=35, edgecolors="none")
    ax.set_xlabel("smooth_mean_abs_diff (lower is better)")
    ax.set_ylabel("corr_mse (higher is better)")
    ax.set_title("Pareto: Corr vs Smooth")
    fig.colorbar(sc, ax=ax, shrink=0.85, label="score_gating")
    fig.tight_layout()
    fig.savefig(pareto2, dpi=200)
    plt.close(fig)

    # append report references + conclusion template
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n## Component Contributions\n\n")
        f.write("- contrib_equal.png\n")
        f.write("- contrib_gating.png\n")
        f.write("- contrib_regime.png\n")
        f.write("\n## Pareto Plots\n\n")
        f.write("- pareto_auc_vs_smooth.png\n")
        f.write("- pareto_corr_vs_smooth.png\n")
        f.write("\n## Conclusion Template\n\n")
        f.write("We first filtered unstable configurations by smoothness and correlation constraints, ")
        f.write("then ranked candidates by gating-friendly score. The top configuration balances ")
        f.write("regime separation (AUC/sep) and prediction consistency (corr_mse) while keeping ")
        f.write("lambda smoothness within acceptable bounds.\n")

    print("=== Step4: rescore configs ===")
    print(f"[OK] Saved: {rescored_csv}")
    print(f"[OK] Saved: {rescored_md}")
    if merged_path:
        print(f"[OK] Saved: {merged_path}")
    if filtered_out:
        print(f"[OK] Saved: {filtered_path}")
    print(f"[OK] Saved: {top_txt}")
    print(f"[OK] Saved: {report_path}")
    print("Top-5 consistency:", sorted(list(common_top5)))


if __name__ == "__main__":
    main()
