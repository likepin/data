import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from step5_utils import (
    load_lambda_and_mask,
    mean_over_mask,
    find_true_change_from_deltaA,
    edges_from_adj,
)
from graph_io import load_adj, assert_orientation
from step5_pred import compute_change_scores, binarize_topk_on_base


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


def write_logs(logs, out_path):
    safe_mkdir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs) + "\n")


def load_config(cfg_path, data_dir):
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # default config
    return {
        "pred_prefix": "cmiknn",
        "score_type": "valdiff",
        "delta_mode": "A1_minus_A0",
        "gate_mode": "soft",
        "tau_hard": 0.8,
        "w_soft": None,
        "subset_high_q": 0.90,
        "subset_low_q": 0.50,
        "edge_mask": "base_only",
        "norm": "none",
        "output_topk_edges": 20,
    }


def load_A_base(data_dir, logs):
    a_base_path = os.path.join(data_dir, "A_base.npy")
    if os.path.isfile(a_base_path):
        A = np.load(a_base_path)
        if A.ndim == 3:
            base_sum = A.sum(axis=0)
            logs.append(f"A_base shape={A.shape} (sum over lags)")
        else:
            base_sum = A.astype(np.float32)
            logs.append(f"A_base shape={A.shape}")
        return base_sum, a_base_path
    adj_base_path = os.path.join(data_dir, "adj_base.npy")
    if os.path.isfile(adj_base_path):
        adj = np.load(adj_base_path).astype(np.float32)
        logs.append(f"adj_base shape={adj.shape}")
        return adj, adj_base_path
    raise FileNotFoundError("A_base.npy or adj_base.npy not found.")


def load_A_regime(data_dir, prefix, logs):
    v0 = os.path.join(data_dir, f"{prefix}_regime0_val_matrix.npy")
    v1 = os.path.join(data_dir, f"{prefix}_regime1_val_matrix.npy")
    a0 = os.path.join(data_dir, f"{prefix}_regime0_adj_hat.npy")
    a1 = os.path.join(data_dir, f"{prefix}_regime1_adj_hat.npy")

    if os.path.isfile(v0) and os.path.isfile(v1):
        val0 = np.load(v0)
        val1 = np.load(v1)
        if val0.ndim != 3 or val1.ndim != 3:
            raise ValueError("val_matrix must be 3D (src,tgt,lag)")
        s0 = best_signed_val_over_lags(val0)  # (src,tgt)
        s1 = best_signed_val_over_lags(val1)  # (src,tgt)
        A0 = s0.T
        A1 = s1.T
        logs.append("A0/A1 loaded from val_matrix")
        return A0, A1
    if os.path.isfile(a0) and os.path.isfile(a1):
        A0 = np.load(a0).astype(np.float32)
        A1 = np.load(a1).astype(np.float32)
        logs.append("A0/A1 loaded from adj_hat")
        return A0, A1
    raise FileNotFoundError(f"prefix {prefix} not found for val_matrix/adj_hat")


def best_signed_val_over_lags(val_matrix):
    vals = val_matrix[:, :, 1:]
    idx = np.argmax(np.abs(vals), axis=2)
    out = np.zeros(vals.shape[:2], dtype=np.float32)
    for src in range(out.shape[0]):
        for tgt in range(out.shape[1]):
            k = int(idx[src, tgt])
            out[src, tgt] = vals[src, tgt, k]
    return out


def subset_masks(lambda_t, valid_mask, high_q, low_q, logs):
    valid_vals = lambda_t[valid_mask]
    valid_vals = valid_vals[np.isfinite(valid_vals)]
    high_thr = float(np.quantile(valid_vals, high_q)) if valid_vals.size > 0 else np.nan
    low_thr = float(np.quantile(valid_vals, low_q)) if valid_vals.size > 0 else np.nan
    high_mask = valid_mask & (lambda_t >= high_thr)
    low_mask = valid_mask & (lambda_t <= low_thr)
    all_mask = valid_mask.copy()
    subset_strategy = "quantile"
    high_non_sat = None
    if valid_vals.size > 0 and (np.all(valid_vals == 1.0) or np.std(valid_vals) < 1e-6):
        non_sat = valid_mask & (lambda_t < 1.0)
        non_vals = lambda_t[non_sat]
        if non_vals.size > 0:
            high_thr = float(np.quantile(non_vals, high_q))
            high_mask = non_sat & (lambda_t >= high_thr)
            subset_strategy = "topk_non_saturated"
            high_non_sat = high_mask.copy()
        else:
            logs.append("WARN: high subset saturated; using top-1 by lambda.")
            idx_max = int(np.argmax(lambda_t))
            high_mask = np.zeros_like(valid_mask, dtype=bool)
            high_mask[idx_max] = True
            subset_strategy = "top1_fallback"
    return high_mask, low_mask, all_mask, high_thr, low_thr, subset_strategy, high_non_sat


def dist_l1(A_eff, A_ref, mask):
    diff = np.abs(A_eff - A_ref)
    diff = diff * mask
    M = max(mask.sum(), 1e-12)
    return float(diff.sum() / M)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    logs = []
    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "exports_step5pp")
    safe_mkdir(out_dir)

    cfg_path = args.config or os.path.join(data_dir, "step5pp_config.json")
    cfg = load_config(cfg_path, data_dir)

    lambda_t, valid_mask, lambda_source, t_switch = load_lambda_and_mask(data_dir, logs)

    A_base, base_path = load_A_base(data_dir, logs)
    assert_orientation(A_base, "tgt_src")
    A0, A1 = load_A_regime(data_dir, cfg.get("pred_prefix", "cmiknn"), logs)
    assert_orientation(A0, "tgt_src")
    assert_orientation(A1, "tgt_src")

    logs.append(f"A_base nnz={(np.abs(A_base) > 0).sum()}")
    logs.append(f"A0 stats: min={A0.min():.4f} max={A0.max():.4f} mean={A0.mean():.4f} nnz={(np.abs(A0) > 0).sum()}")
    logs.append(f"A1 stats: min={A1.min():.4f} max={A1.max():.4f} mean={A1.mean():.4f} nnz={(np.abs(A1) > 0).sum()}")

    if valid_mask.sum() > 0:
        v = lambda_t[valid_mask]
        v = v[np.isfinite(v)]
        if v.size > 0:
            logs.append(f"lambda stats: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

    # true edges from DeltaA
    adj_true, delta_path = find_true_change_from_deltaA(data_dir, logs)
    true_edges = edges_from_adj(adj_true, diag_excluded=True)
    K_true = len(true_edges)

    # delta proxy
    delta_mode = cfg.get("delta_mode", "A1_minus_A0")
    if delta_mode == "A1_minus_A0":
        delta_proxy = A1 - A0
    else:
        delta_proxy = A1 - A0

    delta_mag = np.abs(delta_proxy)

    # edge mask
    edge_mask = np.ones_like(A_base, dtype=np.float32)
    if cfg.get("edge_mask", "base_only") == "base_only":
        edge_mask = (A_base != 0).astype(np.float32)
        np.fill_diagonal(edge_mask, 0.0)
        delta_proxy = delta_proxy * edge_mask
        delta_mag = delta_mag * edge_mask

    top_k = int(cfg.get("output_topk_edges", K_true))
    pred_adj, pred_scores = binarize_topk_on_base(delta_mag, A_base, top_k, diag_excluded=True)

    pred_edges = edges_from_adj(pred_adj, diag_excluded=True)
    pred_csv = os.path.join(out_dir, "pred_topk_edges.csv")
    with open(pred_csv, "w", encoding="utf-8") as f:
        f.write("rank,src,tgt,delta_mag,delta_signed\n")
        edges_list = list(pred_edges)
        edges_list.sort(key=lambda e: pred_scores.get(e, 0.0), reverse=True)
        for i, (src, tgt) in enumerate(edges_list, start=1):
            f.write(f"{i},{src},{tgt},{abs(delta_proxy[tgt, src]):.6f},{delta_proxy[tgt, src]:.6f}\n")

    gate_mode = cfg.get("gate_mode", "soft")
    tau_hard = float(cfg.get("tau_hard", 0.8))
    w_soft = cfg.get("w_soft", None)

    if gate_mode == "soft":
        gate_weight = np.clip(1.0 - lambda_t, 0.0, 1.0)
    else:
        gate_weight = (lambda_t < tau_hard).astype(np.float32)

    # subset masks
    high_mask, low_mask, all_mask, high_thr, low_thr, subset_strategy, high_non_sat = subset_masks(
        lambda_t, valid_mask, cfg.get("subset_high_q", 0.90), cfg.get("subset_low_q", 0.50), logs
    )

    # distance curves
    dist_base = []
    dist_reg0 = []
    dist_reg1 = []
    retained_ratio = []

    raw_strength = np.mean([abs(delta_proxy[tgt, src]) for (src, tgt) in pred_edges]) if pred_edges else 0.0
    eps = 1e-12

    for t in range(len(lambda_t)):
        g = gate_weight[t]
        A_eff = A_base + g * delta_proxy
        dist_base.append(dist_l1(A_eff, A_base, edge_mask))
        dist_reg0.append(dist_l1(A_eff, A0, edge_mask))
        dist_reg1.append(dist_l1(A_eff, A1, edge_mask))
        if pred_edges:
            retained = np.mean([abs(g * delta_proxy[tgt, src]) for (src, tgt) in pred_edges])
        else:
            retained = 0.0
        retained_ratio.append(retained / (raw_strength + eps))

    dist_base = np.array(dist_base, dtype=np.float32)
    dist_reg0 = np.array(dist_reg0, dtype=np.float32)
    dist_reg1 = np.array(dist_reg1, dtype=np.float32)
    retained_ratio = np.array(retained_ratio, dtype=np.float32)

    def subset_row(name, mask):
        return {
            "subset": name,
            "count": int(mask.sum()),
            "mean_lambda": mean_over_mask(lambda_t, mask),
            "mean_gate_weight": mean_over_mask(gate_weight, mask),
            "p_active": float((gate_weight[mask] > (w_soft if w_soft is not None else 0.0)).mean()) if mask.sum() > 0 else 0.0,
            "mean_dist_base": mean_over_mask(dist_base, mask),
            "mean_dist_reg0": mean_over_mask(dist_reg0, mask),
            "mean_dist_reg1": mean_over_mask(dist_reg1, mask),
            "mean_retained_ratio": mean_over_mask(retained_ratio, mask),
        }

    summary_rows = [
        subset_row("high", high_mask),
        subset_row("low", low_mask),
        subset_row("all", all_mask),
    ]
    if high_non_sat is not None:
        summary_rows.append(subset_row("high_non_sat", high_non_sat))

    summary_csv = os.path.join(out_dir, "step5pp_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(",".join(summary_rows[0].keys()) + "\n")
        for r in summary_rows:
            f.write(",".join([str(r[k]) for k in summary_rows[0].keys()]) + "\n")

    summary_md = os.path.join(out_dir, "step5pp_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("## Step5++ Summary\n\n")
        f.write("| " + " | ".join(summary_rows[0].keys()) + " |\n")
        f.write("| " + " | ".join(["---"] * len(summary_rows[0])) + " |\n")
        for r in summary_rows:
            f.write("| " + " | ".join([str(r[k]) for k in summary_rows[0].keys()]) + " |\n")
        f.write("\n")
        f.write("- high should have smallest dist_base\n")
        f.write("- low should have larger dist_base and higher retained_ratio\n")

    retained_csv = os.path.join(out_dir, "retained_summary.csv")
    with open(retained_csv, "w", encoding="utf-8") as f:
        f.write("t,retained_ratio\n")
        for i, v in enumerate(retained_ratio):
            f.write(f"{i},{v:.6f}\n")

    retained_fig = os.path.join(out_dir, "retained_curve.png")
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(retained_ratio, color="tab:purple", linewidth=1.0)
    ax.set_title("retained_ratio(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("ratio")
    fig.tight_layout()
    fig.savefig(retained_fig, dpi=200)
    plt.close(fig)

    sim_fig = os.path.join(out_dir, "gated_graph_simulation.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(lambda_t, color="tab:blue", label="lambda")
    axes[0].plot(gate_weight, color="tab:orange", label="g(t)")
    if t_switch is not None:
        axes[0].axvline(t_switch, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].legend(loc="upper right")
    axes[0].set_title("lambda(t) and gate_weight(t)")

    axes[1].plot(dist_base, label="dist_base")
    axes[1].plot(dist_reg0, label="dist_reg0")
    axes[1].plot(dist_reg1, label="dist_reg1")
    axes[1].legend(loc="upper right")
    axes[1].set_title("distance curves")

    axes[2].plot(retained_ratio, color="tab:purple", label="retained_ratio")
    axes[2].legend(loc="upper right")
    axes[2].set_title("retained_ratio(t)")
    axes[2].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(sim_fig, dpi=200)
    plt.close(fig)

    config_used = {
        "data_dir": data_dir,
        "config_path": cfg_path,
        "lambda_source": lambda_source,
        "base_path": base_path,
        "delta_path": delta_path,
        "pred_prefix": cfg.get("pred_prefix", ""),
        "score_type": cfg.get("score_type", ""),
        "delta_mode": delta_mode,
        "gate_mode": gate_mode,
        "tau_hard": tau_hard,
        "w_soft": w_soft,
        "subset_high_q": cfg.get("subset_high_q", 0.90),
        "subset_low_q": cfg.get("subset_low_q", 0.50),
        "subset_strategy": subset_strategy,
        "high_thr": high_thr,
        "low_thr": low_thr,
        "edge_mask": cfg.get("edge_mask", "base_only"),
        "output_topk_edges": top_k,
        "K_true": K_true,
        "K_pred": len(pred_edges),
    }
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config_used, f, indent=2)

    write_logs(logs, os.path.join(out_dir, "logs.txt"))

    if args.sanity:
        print("=== Step5++ sanity ===")
        print(f"A_base nnz={int((np.abs(A_base) > 0).sum())}")
        print(f"A0 min/max/mean: {A0.min():.4f}/{A0.max():.4f}/{A0.mean():.4f}")
        print(f"A1 min/max/mean: {A1.min():.4f}/{A1.max():.4f}/{A1.mean():.4f}")
        print(f"K_true={K_true}, K_pred={len(pred_edges)}")
        print(f"subset_strategy={subset_strategy} high_thr={high_thr} low_thr={low_thr}")


if __name__ == "__main__":
    main()
