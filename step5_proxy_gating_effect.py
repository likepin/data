import os
import json
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt

from step5_utils import (
    safe_mkdir,
    log_append,
    write_logs,
    load_lambda_and_mask,
    detect_lambda_config,
    find_true_change_adj,
    find_base_adj,
    find_pred_change_adj,
    edges_from_adj,
    confusion,
    compute_expected_metrics,
    quantile_mask,
    mean_over_mask,
    active_fraction_hard,
    active_fraction_soft,
    load_valdiff_scores,
)


def parse_float_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def write_md(rows, out_path, title, columns):
    safe_mkdir(os.path.dirname(out_path))
    lines = [f"## {title}\n",
             "| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for r in rows:
        row_vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                row_vals.append(f"{v:.6f}")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_gating_curve(lambda_t, t_switch, K_pred, tau, out_path):
    T = len(lambda_t)
    x = np.arange(T)
    active = (lambda_t < tau)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(lambda_t, color="tab:blue", linewidth=1.0)
    axes[0].set_title("lambda_t")
    axes[0].set_ylabel("lambda")
    if t_switch is not None:
        axes[0].axvline(t_switch, color="tab:red", linestyle="--", linewidth=1.0, label="t_switch")
        axes[0].legend(loc="upper right")

    hard_count = np.where(active, K_pred, 0)
    soft_count = K_pred * (1.0 - lambda_t)
    axes[1].plot(hard_count, color="tab:orange", linewidth=1.0, label="hard gated")
    axes[1].plot(soft_count, color="tab:green", linewidth=1.0, label="soft gated")
    axes[1].plot(np.full_like(hard_count, K_pred), color="tab:gray", linewidth=1.0, label="ungated")
    axes[1].set_title("active_edge_count(t)")
    axes[1].set_ylabel("count")
    axes[1].set_xlabel("time t")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_top_edges_csv(edges, scores, true_edges, subset_stats, out_path, top_k):
    rows = []
    edges_list = list(edges)
    if scores:
        edges_list.sort(key=lambda e: scores.get(e, 0.0), reverse=True)
    else:
        edges_list.sort()
    top = edges_list[:top_k]
    for i, (src, tgt) in enumerate(top, start=1):
        rows.append({
            "rank": i,
            "src": src,
            "tgt": tgt,
            "score": float(scores.get((src, tgt), 1.0)) if scores else 1.0,
            "is_true_change": 1 if (src, tgt) in true_edges else 0,
            "mean_lambda": subset_stats["mean_lambda"],
            "mean_gate_weight": subset_stats["mean_gate_weight"],
            "count_active": subset_stats["count_active"],
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "rank", "src", "tgt", "score", "is_true_change",
            "mean_lambda", "mean_gate_weight", "count_active"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--high_q", type=float, default=0.90)
    parser.add_argument("--low_q", type=float, default=0.50)
    parser.add_argument("--tau_list", type=str, default="0.7,0.8,0.9")
    parser.add_argument("--soft_w_list", type=str, default="0.2,0.3,0.4")
    parser.add_argument("--soft_mode", type=str, default="mean_w", choices=["mean_w", "frac_active"])
    parser.add_argument("--p_thresh", type=float, default=0.5)
    parser.add_argument("--lambda_config", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    logs = []
    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "exports_step5")
    safe_mkdir(out_dir)

    lambda_t, valid_mask, lambda_source, t_switch = load_lambda_and_mask(data_dir, logs)
    lambda_config = args.lambda_config or detect_lambda_config(data_dir, fallback="(unknown)")

    adj_true, true_path = find_true_change_adj(data_dir, logs)
    _, base_path = find_base_adj(data_dir, logs)
    pred_adj, pred_source, pred_prefix, pred_scores = find_pred_change_adj(data_dir, logs)

    if pred_adj is None:
        raise RuntimeError("pred_adj is None (no predicted change edges found).")

    true_edges = edges_from_adj(adj_true, diag_excluded=True)
    pred_edges = edges_from_adj(pred_adj, diag_excluded=True)
    K_true = len(true_edges)
    K_pred = len(pred_edges)
    delta_source = f"{pred_source.replace(':', '_')}_topK{K_pred}"

    tp0, fp0, fn0, prec0, rec0, f10 = confusion(pred_edges, true_edges)

    tau_list = parse_float_list(args.tau_list)
    soft_w_list = parse_float_list(args.soft_w_list)

    # subset masks
    high_mask = quantile_mask(lambda_t, valid_mask, args.high_q, mode="ge")
    low_mask = quantile_mask(lambda_t, valid_mask, args.low_q, mode="le")
    all_mask = valid_mask.copy()

    subset_info = {
        "high": high_mask,
        "low": low_mask,
        "all": all_mask,
    }

    summary_rows = []
    retention_rows = []

    # ungated baseline (p_active=1)
    for subset_name, mask in subset_info.items():
        tp, fp, fn, prec, rec, f1, shd = compute_expected_metrics(tp0, fp0, K_true, p_active=1.0)
        summary_rows.append({
            "lambda_config": lambda_config,
            "deltaA_source": delta_source,
            "gate_type": "ungated",
            "tau": "",
            "subset": subset_name,
            "subset_q": f"high_q={args.high_q:.2f},low_q={args.low_q:.2f}",
            "K_true_change": K_true,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Prec": prec,
            "Rec": rec,
            "F1": f1,
            "SHD": shd,
            "SHD_gain_vs_ungated": "",
            "F1_delta_vs_ungated": "",
        })
        retention_rows.append({
            "lambda_config": lambda_config,
            "deltaA_source": delta_source,
            "gate_type": "ungated",
            "tau": "",
            "subset": subset_name,
            "K_pred": K_pred,
            "TP_change": tp0,
            "FP_change": fp0,
            "retained_ratio": 1.0,
            "true_retained_ratio": 1.0 if tp0 > 0 else np.nan,
            "fp_removed_ratio": 0.0,
        })

    # hard gate
    for tau in tau_list:
        for subset_name, mask in subset_info.items():
            p_active, count_active = active_fraction_hard(lambda_t, mask, tau)
            tp, fp, fn, prec, rec, f1, shd = compute_expected_metrics(tp0, fp0, K_true, p_active=p_active)
            summary_rows.append({
                "lambda_config": lambda_config,
                "deltaA_source": delta_source,
                "gate_type": "hard",
                "tau": tau,
                "subset": subset_name,
                "subset_q": f"high_q={args.high_q:.2f},low_q={args.low_q:.2f}",
                "K_true_change": K_true,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Prec": prec,
                "Rec": rec,
                "F1": f1,
                "SHD": shd,
                "SHD_gain_vs_ungated": "",
                "F1_delta_vs_ungated": "",
            })
            retention_rows.append({
                "lambda_config": lambda_config,
                "deltaA_source": delta_source,
                "gate_type": "hard",
                "tau": tau,
                "subset": subset_name,
                "K_pred": K_pred,
                "TP_change": tp0 * p_active,
                "FP_change": fp0 * p_active,
                "retained_ratio": p_active,
                "true_retained_ratio": (p_active if tp0 > 0 else np.nan),
                "fp_removed_ratio": 1.0 - p_active,
            })

    # soft gate
    for w_thresh in soft_w_list:
        for subset_name, mask in subset_info.items():
            p_active = active_fraction_soft(
                lambda_t, mask, w_thresh,
                mode=args.soft_mode,
                p_thresh=args.p_thresh
            )
            tp, fp, fn, prec, rec, f1, shd = compute_expected_metrics(tp0, fp0, K_true, p_active=p_active)
            summary_rows.append({
                "lambda_config": lambda_config,
                "deltaA_source": delta_source,
                "gate_type": "soft",
                "tau": w_thresh,
                "subset": subset_name,
                "subset_q": f"high_q={args.high_q:.2f},low_q={args.low_q:.2f}",
                "K_true_change": K_true,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Prec": prec,
                "Rec": rec,
                "F1": f1,
                "SHD": shd,
                "SHD_gain_vs_ungated": "",
                "F1_delta_vs_ungated": "",
            })
            retention_rows.append({
                "lambda_config": lambda_config,
                "deltaA_source": delta_source,
                "gate_type": "soft",
                "tau": w_thresh,
                "subset": subset_name,
                "K_pred": K_pred,
                "TP_change": tp0 * p_active,
                "FP_change": fp0 * p_active,
                "retained_ratio": p_active,
                "true_retained_ratio": (p_active if tp0 > 0 else np.nan),
                "fp_removed_ratio": 1.0 - p_active,
            })

    # fill SHD_gain vs ungated and F1_delta vs ungated
    def key(r):
        return (r["gate_type"], r["tau"], r["subset"])

    ungated_map = {r["subset"]: r for r in summary_rows if r["gate_type"] == "ungated"}
    for r in summary_rows:
        if r["gate_type"] == "ungated":
            continue
        base = ungated_map.get(r["subset"])
        if base is None:
            continue
        if r["subset"] in ("high", "all"):
            r["SHD_gain_vs_ungated"] = float(base["SHD"] - r["SHD"])
        if r["subset"] in ("low", "all"):
            r["F1_delta_vs_ungated"] = float(r["F1"] - base["F1"])

    # top edges
    scores = pred_scores
    if scores is None and pred_prefix:
        scores = load_valdiff_scores(data_dir, pred_prefix, logs)
    top_k = args.top_k or K_true

    mean_lambda_high = mean_over_mask(lambda_t, high_mask)
    mean_lambda_low = mean_over_mask(lambda_t, low_mask)
    mean_gate_high = mean_over_mask(1.0 - lambda_t, high_mask)
    mean_gate_low = mean_over_mask(1.0 - lambda_t, low_mask)

    # use first tau for hard gating stats
    tau_demo = tau_list[0] if tau_list else 0.8
    _, count_active_high = active_fraction_hard(lambda_t, high_mask, tau_demo)
    _, count_active_low = active_fraction_hard(lambda_t, low_mask, tau_demo)

    build_top_edges_csv(
        pred_edges, scores, true_edges,
        {
            "mean_lambda": mean_lambda_high,
            "mean_gate_weight": mean_gate_high,
            "count_active": count_active_high,
        },
        os.path.join(out_dir, "top_edges_highlambda.csv"),
        top_k
    )
    build_top_edges_csv(
        pred_edges, scores, true_edges,
        {
            "mean_lambda": mean_lambda_low,
            "mean_gate_weight": mean_gate_low,
            "count_active": count_active_low,
        },
        os.path.join(out_dir, "top_edges_lowlambda.csv"),
        top_k
    )

    # gating curve demo
    gating_curve_path = os.path.join(out_dir, "gating_curve_demo.png")
    plot_gating_curve(lambda_t, t_switch, K_pred, tau_demo, gating_curve_path)

    # outputs
    summary_csv = os.path.join(out_dir, "step5_proxy_summary.csv")
    retention_csv = os.path.join(out_dir, "step5_edge_retention.csv")
    summary_md = os.path.join(out_dir, "step5_proxy_summary.md")
    retention_md = os.path.join(out_dir, "step5_edge_retention.md")

    summary_cols = [
        "lambda_config", "deltaA_source", "gate_type", "tau", "subset", "subset_q",
        "K_true_change", "TP", "FP", "FN", "Prec", "Rec", "F1", "SHD",
        "SHD_gain_vs_ungated", "F1_delta_vs_ungated"
    ]
    retention_cols = [
        "lambda_config", "deltaA_source", "gate_type", "tau", "subset",
        "K_pred", "TP_change", "FP_change", "retained_ratio",
        "true_retained_ratio", "fp_removed_ratio"
    ]

    # write CSV/MD
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in summary_cols})
    write_md(summary_rows, summary_md, "Table 5-1: Proxy Summary", summary_cols)

    with open(retention_csv, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=retention_cols)
        w.writeheader()
        for r in retention_rows:
            w.writerow({k: r.get(k, "") for k in retention_cols})
    write_md(retention_rows, retention_md, "Table 5-3: Edge Retention", retention_cols)

    config_used = {
        "data_dir": data_dir,
        "lambda_source": lambda_source,
        "lambda_config": lambda_config,
        "t_switch": t_switch,
        "true_change_path": true_path,
        "base_path": base_path,
        "pred_source": pred_source,
        "deltaA_source": delta_source,
        "K_true": K_true,
        "K_pred": K_pred,
        "high_q": args.high_q,
        "low_q": args.low_q,
        "tau_list": tau_list,
        "soft_w_list": soft_w_list,
        "soft_mode": args.soft_mode,
        "p_thresh": args.p_thresh,
        "top_k": top_k,
    }
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config_used, f, indent=2)

    write_logs(logs, os.path.join(out_dir, "logs.txt"))

    print("=== Step5 proxy gating effect ===")
    print(f"[OK] Summary: {summary_csv}")
    print(f"[OK] Retention: {retention_csv}")
    print(f"[OK] Top edges: top_edges_highlambda.csv / top_edges_lowlambda.csv")
    print(f"[OK] Curve: {gating_curve_path}")


if __name__ == "__main__":
    main()
