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
    read_json,
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
    find_true_change_from_deltaA,
    gated_change_from_deltaA,
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
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--high_q", type=float, default=0.90)
    parser.add_argument("--low_q", type=float, default=0.50)
    parser.add_argument("--tau_list", type=str, default="0.7,0.8,0.9")
    parser.add_argument("--soft_w_list", type=str, default="0.2,0.3,0.4")
    parser.add_argument("--soft_mode", type=str, default="mean_w", choices=["mean_w", "frac_active"])
    parser.add_argument("--p_thresh", type=float, default=0.5)
    parser.add_argument("--lambda_config", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    logs = []
    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "exports_step5")
    safe_mkdir(out_dir)

    cfg_path = args.config or os.path.join(data_dir, "step5_config.json")
    if not os.path.isfile(cfg_path):
        alt_cfg = "step5_config.json"
        if os.path.isfile(alt_cfg):
            cfg_path = alt_cfg
            log_append(logs, f"WARN: using config from CWD: {cfg_path}")
        else:
            raise FileNotFoundError(f"step5_config.json not found: {cfg_path} (and no ./step5_config.json)")
    cfg = read_json(cfg_path)

    lambda_t, valid_mask, lambda_source, t_switch = load_lambda_and_mask(data_dir, logs)
    lambda_config = args.lambda_config or detect_lambda_config(data_dir, fallback="(unknown)")

    adj_true, true_path = find_true_change_adj(data_dir, logs)
    adj_base, base_path = find_base_adj(data_dir, logs)
    a_base_path = os.path.join(data_dir, "A_base.npy")
    A_base = np.load(a_base_path) if os.path.isfile(a_base_path) else None
    pred_adj, pred_source, pred_prefix, pred_scores = find_pred_change_adj(data_dir, cfg, logs)

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

    # subset masks with saturation handling
    valid_vals = lambda_t[valid_mask]
    valid_vals = valid_vals[np.isfinite(valid_vals)]
    high_thr = float(np.quantile(valid_vals, args.high_q)) if valid_vals.size > 0 else np.nan
    low_thr = float(np.quantile(valid_vals, args.low_q)) if valid_vals.size > 0 else np.nan
    high_mask = quantile_mask(lambda_t, valid_mask, args.high_q, mode="ge")
    low_mask = quantile_mask(lambda_t, valid_mask, args.low_q, mode="le")
    all_mask = valid_mask.copy()
    subset_strategy = "quantile"
    if valid_vals.size > 0:
        if np.all(valid_vals == 1.0) or np.std(valid_vals) < 1e-6:
            for q in (0.95, 0.98):
                high_thr = float(np.quantile(valid_vals, q))
                high_mask = valid_mask & (lambda_t >= high_thr)
                if high_mask.sum() > 0:
                    subset_strategy = f"quantile_{q:.2f}"
                    break
            if high_mask.sum() == 0:
                non_sat = valid_mask & (lambda_t < 1.0)
                non_vals = lambda_t[non_sat]
                if non_vals.size > 0:
                    high_thr = float(np.quantile(non_vals, args.high_q))
                    high_mask = non_sat & (lambda_t >= high_thr)
                    subset_strategy = "topk_non_saturated"
                else:
                    log_append(logs, "WARN: high subset saturated; using top-1 by lambda.")
                    idx_max = int(np.argmax(lambda_t))
                    high_mask = np.zeros_like(valid_mask, dtype=bool)
                    high_mask[idx_max] = True
                    subset_strategy = "top1_fallback"

    subset_info = {
        "high": high_mask,
        "low": low_mask,
        "all": all_mask,
    }

    mean_lambda_high = mean_over_mask(lambda_t, high_mask)
    mean_lambda_low = mean_over_mask(lambda_t, low_mask)
    mean_lambda_all = mean_over_mask(lambda_t, all_mask)
    mean_gate_high = mean_over_mask(1.0 - lambda_t, high_mask)
    mean_gate_low = mean_over_mask(1.0 - lambda_t, low_mask)
    mean_gate_all = mean_over_mask(1.0 - lambda_t, all_mask)
    log_append(logs, f"mean_lambda_high={mean_lambda_high:.6f} mean_lambda_low={mean_lambda_low:.6f} mean_lambda_all={mean_lambda_all:.6f}")
    log_append(logs, f"mean_gate_high={mean_gate_high:.6f} mean_gate_low={mean_gate_low:.6f} mean_gate_all={mean_gate_all:.6f}")
    if tau_list:
        p_active_high = float((lambda_t[high_mask] < tau_list[0]).mean()) if high_mask.sum() > 0 else 0.0
        p_active_low = float((lambda_t[low_mask] < tau_list[0]).mean()) if low_mask.sum() > 0 else 0.0
        log_append(logs, f"p_active_high={p_active_high:.6f} p_active_low={p_active_low:.6f} tau0={tau_list[0]}")

    summary_rows = []
    retention_rows = []

    def safe_ratio(num, den):
        den_is_zero = False
        if den <= 0:
            den_is_zero = True
            return 0.0, den_is_zero
        return float(num / den), den_is_zero

    def mean_gate_weight_hard(mask, tau):
        vals = lambda_t[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0
        return float((vals < tau).mean())

    def mean_gate_weight_soft(mask):
        vals = lambda_t[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0
        return float(np.clip(1.0 - vals, 0.0, 1.0).mean())

    # ungated baseline (p_active=1)
    for subset_name, mask in subset_info.items():
        tp, fp, fn, prec, rec, f1, shd = compute_expected_metrics(tp0, fp0, K_true, p_active=1.0)
        mean_lambda_subset = mean_over_mask(lambda_t, mask)
        mean_gate_weight_subset = 1.0
        if A_base is not None and os.path.isfile(os.path.join(data_dir, "DeltaA.npy")):
            DeltaA = np.load(os.path.join(data_dir, "DeltaA.npy"))
            change_gated = gated_change_from_deltaA(A_base, DeltaA, gate_weight=1.0)
            edges_gated = edges_from_adj(change_gated, diag_excluded=True)
            rt_tp, rt_fp, rt_fn, rt_prec, rt_rec, rt_f1 = confusion(edges_gated, true_edges)
            rt_shd = rt_fp + rt_fn
        else:
            rt_tp = rt_fp = rt_fn = rt_prec = rt_rec = rt_f1 = rt_shd = np.nan
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
            "mean_lambda_subset": mean_lambda_subset,
            "mean_gate_weight_subset": mean_gate_weight_subset,
            "real_TP": rt_tp,
            "real_FP": rt_fp,
            "real_FN": rt_fn,
            "real_Prec": rt_prec,
            "real_Rec": rt_rec,
            "real_F1": rt_f1,
            "real_SHD": rt_shd,
        })
        retained_ratio, retained_den_zero = safe_ratio(K_pred, K_pred)
        n_true_retained = tp0
        true_retained_ratio, true_den_zero = safe_ratio(n_true_retained, K_true)
        n_fp_removed = 0.0
        fp_removed_ratio, fp_den_zero = safe_ratio(n_fp_removed, fp0)
        retention_rows.append({
            "lambda_config": lambda_config,
            "deltaA_source": delta_source,
            "gate_type": "ungated",
            "tau": "",
            "subset": subset_name,
            "K_pred": K_pred,
            "TP_change": tp0,
            "FP_change": fp0,
            "retained_ratio": retained_ratio,
            "true_retained_ratio": true_retained_ratio,
            "fp_removed_ratio": fp_removed_ratio,
            "retained_den_zero": retained_den_zero,
            "true_retained_den_zero": true_den_zero,
            "fp_removed_den_zero": fp_den_zero,
            "n_true_edges_subset": K_true,
            "n_true_retained": n_true_retained,
            "n_fp_edges_subset": fp0,
            "n_fp_removed": n_fp_removed,
            "n_pred_edges": K_pred,
            "n_true_edges": K_true,
            "n_active_edges_high": K_pred,
            "n_active_edges_low": K_pred,
        })

    # hard gate
    for tau in tau_list:
        p_active_high, _ = active_fraction_hard(lambda_t, high_mask, tau)
        p_active_low, _ = active_fraction_hard(lambda_t, low_mask, tau)
        n_active_high = K_pred * p_active_high
        n_active_low = K_pred * p_active_low
        for subset_name, mask in subset_info.items():
            p_active, count_active = active_fraction_hard(lambda_t, mask, tau)
            tp, fp, fn, prec, rec, f1, shd = compute_expected_metrics(tp0, fp0, K_true, p_active=p_active)
            mean_lambda_subset = mean_over_mask(lambda_t, mask)
            mean_gate_weight_subset = mean_gate_weight_hard(mask, tau)
            if A_base is not None and os.path.isfile(os.path.join(data_dir, "DeltaA.npy")):
                DeltaA = np.load(os.path.join(data_dir, "DeltaA.npy"))
                change_gated = gated_change_from_deltaA(A_base, DeltaA, gate_weight=mean_gate_weight_subset)
                edges_gated = edges_from_adj(change_gated, diag_excluded=True)
                rt_tp, rt_fp, rt_fn, rt_prec, rt_rec, rt_f1 = confusion(edges_gated, true_edges)
                rt_shd = rt_fp + rt_fn
            else:
                rt_tp = rt_fp = rt_fn = rt_prec = rt_rec = rt_f1 = rt_shd = np.nan
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
                "mean_lambda_subset": mean_lambda_subset,
                "mean_gate_weight_subset": mean_gate_weight_subset,
                "real_TP": rt_tp,
                "real_FP": rt_fp,
                "real_FN": rt_fn,
                "real_Prec": rt_prec,
                "real_Rec": rt_rec,
                "real_F1": rt_f1,
                "real_SHD": rt_shd,
            })
            retained_ratio, retained_den_zero = safe_ratio(K_pred * p_active, K_pred)
            n_true_retained = tp0 * p_active
            true_retained_ratio, true_den_zero = safe_ratio(n_true_retained, K_true)
            n_fp_removed = fp0 - fp0 * p_active
            fp_removed_ratio, fp_den_zero = safe_ratio(n_fp_removed, fp0)
            retention_rows.append({
                "lambda_config": lambda_config,
                "deltaA_source": delta_source,
                "gate_type": "hard",
                "tau": tau,
                "subset": subset_name,
                "K_pred": K_pred,
                "TP_change": tp0 * p_active,
                "FP_change": fp0 * p_active,
                "retained_ratio": retained_ratio,
                "true_retained_ratio": true_retained_ratio,
                "fp_removed_ratio": fp_removed_ratio,
                "retained_den_zero": retained_den_zero,
                "true_retained_den_zero": true_den_zero,
                "fp_removed_den_zero": fp_den_zero,
                "n_true_edges_subset": K_true,
                "n_true_retained": n_true_retained,
                "n_fp_edges_subset": fp0,
                "n_fp_removed": n_fp_removed,
                "n_pred_edges": K_pred,
                "n_true_edges": K_true,
                "n_active_edges_high": n_active_high,
                "n_active_edges_low": n_active_low,
            })

    # soft gate
    for w_thresh in soft_w_list:
        p_active_high = active_fraction_soft(
            lambda_t, high_mask, w_thresh, mode=args.soft_mode, p_thresh=args.p_thresh
        )
        p_active_low = active_fraction_soft(
            lambda_t, low_mask, w_thresh, mode=args.soft_mode, p_thresh=args.p_thresh
        )
        n_active_high = K_pred * p_active_high
        n_active_low = K_pred * p_active_low
        for subset_name, mask in subset_info.items():
            p_active = active_fraction_soft(
                lambda_t, mask, w_thresh,
                mode=args.soft_mode,
                p_thresh=args.p_thresh
            )
            tp, fp, fn, prec, rec, f1, shd = compute_expected_metrics(tp0, fp0, K_true, p_active=p_active)
            mean_lambda_subset = mean_over_mask(lambda_t, mask)
            mean_gate_weight_subset = mean_gate_weight_soft(mask)
            if A_base is not None and os.path.isfile(os.path.join(data_dir, "DeltaA.npy")):
                DeltaA = np.load(os.path.join(data_dir, "DeltaA.npy"))
                change_gated = gated_change_from_deltaA(A_base, DeltaA, gate_weight=mean_gate_weight_subset)
                edges_gated = edges_from_adj(change_gated, diag_excluded=True)
                rt_tp, rt_fp, rt_fn, rt_prec, rt_rec, rt_f1 = confusion(edges_gated, true_edges)
                rt_shd = rt_fp + rt_fn
            else:
                rt_tp = rt_fp = rt_fn = rt_prec = rt_rec = rt_f1 = rt_shd = np.nan
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
                "mean_lambda_subset": mean_lambda_subset,
                "mean_gate_weight_subset": mean_gate_weight_subset,
                "real_TP": rt_tp,
                "real_FP": rt_fp,
                "real_FN": rt_fn,
                "real_Prec": rt_prec,
                "real_Rec": rt_rec,
                "real_F1": rt_f1,
                "real_SHD": rt_shd,
            })
            retained_ratio, retained_den_zero = safe_ratio(K_pred * p_active, K_pred)
            n_true_retained = tp0 * p_active
            true_retained_ratio, true_den_zero = safe_ratio(n_true_retained, K_true)
            n_fp_removed = fp0 - fp0 * p_active
            fp_removed_ratio, fp_den_zero = safe_ratio(n_fp_removed, fp0)
            retention_rows.append({
                "lambda_config": lambda_config,
                "deltaA_source": delta_source,
                "gate_type": "soft",
                "tau": w_thresh,
                "subset": subset_name,
                "K_pred": K_pred,
                "TP_change": tp0 * p_active,
                "FP_change": fp0 * p_active,
                "retained_ratio": retained_ratio,
                "true_retained_ratio": true_retained_ratio,
                "fp_removed_ratio": fp_removed_ratio,
                "retained_den_zero": retained_den_zero,
                "true_retained_den_zero": true_den_zero,
                "fp_removed_den_zero": fp_den_zero,
                "n_true_edges_subset": K_true,
                "n_true_retained": n_true_retained,
                "n_fp_edges_subset": fp0,
                "n_fp_removed": n_fp_removed,
                "n_pred_edges": K_pred,
                "n_true_edges": K_true,
                "n_active_edges_high": n_active_high,
                "n_active_edges_low": n_active_low,
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
        "SHD_gain_vs_ungated", "F1_delta_vs_ungated",
        "mean_lambda_subset", "mean_gate_weight_subset",
        "real_TP", "real_FP", "real_FN", "real_Prec", "real_Rec", "real_F1", "real_SHD"
    ]
    retention_cols = [
        "lambda_config", "deltaA_source", "gate_type", "tau", "subset",
        "K_pred", "TP_change", "FP_change", "retained_ratio",
        "true_retained_ratio", "fp_removed_ratio",
        "retained_den_zero", "true_retained_den_zero", "fp_removed_den_zero",
        "n_true_edges_subset", "n_true_retained", "n_fp_edges_subset", "n_fp_removed",
        "n_pred_edges", "n_true_edges", "n_active_edges_high", "n_active_edges_low"
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
        "config_path": cfg_path,
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
        "subset_strategy": subset_strategy,
        "high_thr": high_thr,
        "low_thr": low_thr,
    }
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config_used, f, indent=2)

    write_logs(logs, os.path.join(out_dir, "logs.txt"))

    if args.sanity:
        print("=== SANITY MODE ===")
        delta_path = os.path.join(data_dir, "DeltaA.npy")
        if os.path.isfile(delta_path):
            DeltaA = np.load(delta_path)
            print(f"DeltaA shape={DeltaA.shape}")
            true_edges = edges_from_adj(adj_true, diag_excluded=True)
            print(f"K_true_from_DeltaA={len(true_edges)}")
            if adj_base is not None:
                violations = [(s, t) for (s, t) in true_edges if adj_base[t, s] == 0]
                print(f"violations={len(violations)}")
        print(f"K_true={K_true}, len(true_edges)={len(true_edges)}")
        true_list = list(true_edges)[:20]
        print(f"true_edges (first 20): {true_list}")
        print(f"K_pred={K_pred}, len(pred_edges)={len(pred_edges)}")
        pred_list = list(pred_edges)[:20]
        print(f"pred_edges (first 20): {pred_list}")

        # orientation check
        rng = np.random.RandomState(0)
        edges_list = list(pred_edges)
        if edges_list:
            sample = rng.choice(len(edges_list), size=min(5, len(edges_list)), replace=False)
            for idx in sample:
                src, tgt = edges_list[idx]
                ok = int(pred_adj[tgt, src]) == 1
                print(f"orientation check: {src}->{tgt} adj[tgt,src]={pred_adj[tgt, src]} ok={ok}")

        # subset stats
        def subset_stats(name, mask):
            count = int(mask.sum())
            mean_lambda = mean_over_mask(lambda_t, mask)
            print(f"{name}: count={count} mean_lambda={mean_lambda:.6f} high_thr={high_thr:.6f} low_thr={low_thr:.6f}")

        subset_stats("high", high_mask)
        subset_stats("low", low_mask)
        subset_stats("all", all_mask)

        mean_lambda_high = mean_over_mask(lambda_t, high_mask)
        mean_lambda_low = mean_over_mask(lambda_t, low_mask)
        mean_gate_high = mean_over_mask(1.0 - lambda_t, high_mask)
        mean_gate_low = mean_over_mask(1.0 - lambda_t, low_mask)
        p_active_high = float((lambda_t[high_mask] < tau_list[0]).mean()) if tau_list and high_mask.sum() > 0 else 0.0
        p_active_low = float((lambda_t[low_mask] < tau_list[0]).mean()) if tau_list and low_mask.sum() > 0 else 0.0
        print(f"mean_lambda_high={mean_lambda_high:.6f} mean_lambda_low={mean_lambda_low:.6f}")
        print(f"mean_gate_high={mean_gate_high:.6f} mean_gate_low={mean_gate_low:.6f}")
        print(f"p_active_high={p_active_high:.6f} p_active_low={p_active_low:.6f}")

        print("ratio sanity per setting:")
        for r in retention_rows:
            print(
                f"[{r['gate_type']}] tau={r['tau']} subset={r['subset']} "
                f"n_true_edges_subset={r['n_true_edges_subset']} n_true_retained={r['n_true_retained']} "
                f"true_retained_ratio={r['true_retained_ratio']:.6f} den_zero={r['true_retained_den_zero']} | "
                f"n_fp_edges_subset={r['n_fp_edges_subset']} n_fp_removed={r['n_fp_removed']} "
                f"fp_removed_ratio={r['fp_removed_ratio']:.6f} den_zero={r['fp_removed_den_zero']}"
            )

    print("=== Step5 proxy gating effect ===")
    print(f"[OK] Summary: {summary_csv}")
    print(f"[OK] Retention: {retention_csv}")
    print(f"[OK] Top edges: top_edges_highlambda.csv / top_edges_lowlambda.csv")
    print(f"[OK] Curve: {gating_curve_path}")


if __name__ == "__main__":
    main()
