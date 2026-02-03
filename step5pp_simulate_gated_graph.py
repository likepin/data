import os
import json
import argparse

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from step5_utils import (
    load_lambda_and_mask,
    mean_over_mask,
    find_true_change_from_deltaA,
    edges_from_adj,
)
from graph_io import load_adj, assert_orientation
from step5_pred import compute_change_scores, binarize_topk_on_base
from step5pp_utils import compute_lambda_kmeans, pick_lambda_configs_from_step4


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


def load_A0_A1_gt(data_dir, logs):
    A_base, base_path = load_A_base(data_dir, logs)
    delta_path = os.path.join(data_dir, "DeltaA.npy")
    if not os.path.isfile(delta_path):
        raise FileNotFoundError("DeltaA.npy not found for GT regime reference.")
    DeltaA = np.load(delta_path)
    if DeltaA.ndim != 3:
        raise ValueError("DeltaA must be (L,N,N)")
    DeltaA_agg = DeltaA.sum(axis=0)
    A0 = A_base.copy()
    A1 = A0 + DeltaA_agg
    logs.append("A0/A1 loaded from GT (A_base + DeltaA)")
    return A0, A1


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


def subset_masks(lambda_t, valid_mask, high_q, low_q, logs, eps=1e-6):
    valid_vals = lambda_t[valid_mask]
    valid_vals = valid_vals[np.isfinite(valid_vals)]
    low_thr = float(np.quantile(valid_vals, low_q)) if valid_vals.size > 0 else np.nan
    low_mask = valid_mask & (lambda_t <= low_thr)
    all_mask = valid_mask.copy()

    subset_strategy = "non_sat_quantile"
    high_non_sat = None
    high_thr = np.nan
    high_sat = None

    non_sat = valid_mask & (lambda_t < 1.0 - eps)
    non_vals = lambda_t[non_sat]
    if non_vals.size > 0:
        high_thr = float(np.quantile(non_vals, high_q))
        high_non_sat = non_sat & (lambda_t >= high_thr)
    else:
        logs.append("WARN: non-saturated set empty; fallback to high on full valid.")
        high_thr = float(np.quantile(valid_vals, high_q)) if valid_vals.size > 0 else np.nan
        high_sat = valid_mask & (lambda_t >= high_thr)
        subset_strategy = "fallback_quantile"
        if high_sat.sum() == 0:
            logs.append("WARN: high subset saturated; using top-1 by lambda.")
            idx_max = int(np.argmax(lambda_t))
            high_sat = np.zeros_like(valid_mask, dtype=bool)
            high_sat[idx_max] = True
            subset_strategy = "top1_fallback"
        high_non_sat = None

    if high_sat is None:
        high_thr_sat = float(np.quantile(valid_vals, high_q)) if valid_vals.size > 0 else np.nan
        high_sat = valid_mask & (lambda_t >= high_thr_sat)

    return high_non_sat, high_sat, low_mask, all_mask, high_thr, low_thr, subset_strategy


def dist_l1(A_eff, A_ref, mask):
    diff = np.abs(A_eff - A_ref)
    diff = diff * mask
    M = max(mask.sum(), 1e-12)
    return float(diff.sum() / M)


def corrcoef_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return 0.0
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_mask(mode, base_mask, pred_mask):
    if mode == "base_only":
        return base_mask
    if mode == "union_base_predchange":
        return ((base_mask != 0) | (pred_mask != 0)).astype(np.float32)
    if mode == "full_offdiag":
        mask = np.ones_like(base_mask, dtype=np.float32)
        np.fill_diagonal(mask, 0.0)
        return mask
    raise ValueError(f"Unknown mask mode: {mode}")


def dist_mask_union_delta_topk(A0_eff, A1_eff, union_mask, topk=6, thr=None):
    delta_ref = np.abs(A1_eff - A0_eff)
    cand = union_mask != 0
    if not cand.any():
        return np.zeros_like(union_mask, dtype=np.float32)
    vals = delta_ref[cand].reshape(-1)
    if thr is None:
        k = min(int(topk), vals.size)
        if k <= 0:
            return np.zeros_like(union_mask, dtype=np.float32)
        thresh = np.partition(vals, -k)[-k]
    else:
        thresh = float(thr)
    mask = cand & (delta_ref >= thresh)
    return mask.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--topk_mode", type=str, default="match_true", choices=["match_true", "fixed"])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--score_type", type=str, default=None,
                        choices=["score_gating", "score_regime", "score_equal"])
    parser.add_argument("--top_m", type=int, default=5)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    logs = []
    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "exports_step5pp")
    safe_mkdir(out_dir)

    cfg_path = args.config or os.path.join(data_dir, "step5pp_config.json")
    cfg = load_config(cfg_path, data_dir)
    meta_path = os.path.join(data_dir, "meta.json")
    t_switch = None
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        t_switch = int(meta.get("t_switch", 0)) if "t_switch" in meta else None

    A_base, base_path = load_A_base(data_dir, logs)
    assert_orientation(A_base, "tgt_src")
    regime_ref_source = cfg.get("regime_ref_source", "estimated")
    if regime_ref_source == "gt":
        A0, A1 = load_A0_A1_gt(data_dir, logs)
    else:
        A0, A1 = load_A_regime(data_dir, cfg.get("pred_prefix", "cmiknn"), logs)
    assert_orientation(A0, "tgt_src")
    assert_orientation(A1, "tgt_src")

    logs.append(f"A_base nnz={(np.abs(A_base) > 0).sum()}")
    logs.append(f"A0 stats: min={A0.min():.4f} max={A0.max():.4f} mean={A0.mean():.4f} nnz={(np.abs(A0) > 0).sum()}")
    logs.append(f"A1 stats: min={A1.min():.4f} max={A1.max():.4f} mean={A1.mean():.4f} nnz={(np.abs(A1) > 0).sum()}")

    # lambda stats only when available (single-config path)

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

    # topK selection
    topk_mode = cfg.get("topk_mode", args.topk_mode)
    if topk_mode == "match_true":
        top_k = K_true
        k_source = "K_true_from_DeltaA"
    else:
        top_k = int(cfg.get("top_k", args.top_k))
        k_source = "fixed"
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

    def run_once(lambda_t, valid_mask, tag, write_plots):
        if gate_mode == "soft":
            gate_weight = np.clip(1.0 - lambda_t, 0.0, 1.0)
        else:
            gate_weight = (lambda_t < tau_hard).astype(np.float32)
        gate_weight = np.where(valid_mask, gate_weight, 0.0)

        # subset masks
        high_non_sat, high_sat, low_mask, all_mask, high_thr, low_thr, subset_strategy = subset_masks(
            lambda_t, valid_mask, cfg.get("subset_high_q", 0.90), cfg.get("subset_low_q", 0.50), logs
        )

        # masks
        base_mask = (A_base != 0).astype(np.float32)
        pred_mask = (pred_adj != 0).astype(np.float32)
        true_mask = (adj_true != 0).astype(np.float32)
        delta_mask_mode = cfg.get("delta_mask", "union_base_predchange")
        dist_mask_mode = cfg.get("dist_mask", cfg.get("dist_mask_mode", "union_base_predchange"))
        delta_mask = build_mask(delta_mask_mode, base_mask, pred_mask)

        # regime support mask (needed before union_delta_topk)
        regime_support_mode = cfg.get("regime_support_mode", "union_base_predchange")
        if regime_support_mode == "base_only":
            support_mask = base_mask
        elif regime_support_mode == "union_base_predchange":
            support_mask = build_mask("union_base_predchange", base_mask, pred_mask)
        elif regime_support_mode == "true_change_only":
            support_mask = true_mask
        elif regime_support_mode == "pred_change_only":
            support_mask = pred_mask
        else:
            raise ValueError(f"Unknown regime_support_mode: {regime_support_mode}")

        # apply delta mask and regime support
        delta_proxy_masked = delta_proxy * delta_mask
        delta_mag_masked = np.abs(delta_proxy_masked)
        A0_eff = A0 * support_mask
        A1_eff = A1 * support_mask

        # dist mask after A0_eff/A1_eff defined
        if dist_mask_mode == "change_edges_focus":
            focus = pred_mask if pred_edges else true_mask
            dist_mask = (focus != 0).astype(np.float32)
        else:
            if dist_mask_mode == "true_change_only":
                dist_mask = (true_mask != 0).astype(np.float32)
            elif dist_mask_mode == "union_delta_topk":
                union_mask = build_mask("union_base_predchange", base_mask, pred_mask)
                dist_topk = int(cfg.get("dist_topk", 6))
                dist_thr = cfg.get("dist_delta_thr", None)
                dist_mask = dist_mask_union_delta_topk(A0_eff, A1_eff, union_mask, topk=dist_topk, thr=dist_thr)
            else:
                dist_mask = build_mask(dist_mask_mode, base_mask, pred_mask)

        # sanity: mask stats
        if args.sanity:
            nnz_dist = int((dist_mask != 0).sum())
            nnz_delta = int((delta_mask != 0).sum())
            nnz_a0 = int((A0_eff != 0).sum())
            nnz_a1 = int((A1_eff != 0).sum())
            diff = (A1_eff - A0_eff) * dist_mask
            diff_abs = np.abs(diff)
            mean_diff = float(diff_abs[dist_mask != 0].mean()) if nnz_dist > 0 else 0.0
            max_diff = float(diff_abs.max()) if nnz_dist > 0 else 0.0
            print(f"mask nnz: delta_mask={nnz_delta} dist_mask={nnz_dist} A0_eff={nnz_a0} A1_eff={nnz_a1}")
            print(f"|A1_eff-A0_eff| on mask: mean={mean_diff:.6f} max={max_diff:.6f}")
            if nnz_dist > 0:
                assert not np.allclose(A0_eff[dist_mask != 0], A1_eff[dist_mask != 0])
            if dist_mask_mode in ("change_edges_focus", "true_change_only"):
                assert nnz_dist <= max(20, int(0.1 * dist_mask.size))

        # distance curves
        dist_base = []
        dist_reg0 = []
        dist_reg1 = []
        retained_ratio = []

        raw_strength = np.mean([abs(delta_proxy_masked[tgt, src]) for (src, tgt) in pred_edges]) if pred_edges else 0.0
        delta_mag_mean = float(delta_mag_masked[delta_mask > 0].mean()) if delta_mask.sum() > 0 else 0.0
        delta_mag_max = float(delta_mag_masked[delta_mask > 0].max()) if delta_mask.sum() > 0 else 0.0
        eps = 1e-12

        eff_anchor = cfg.get("eff_anchor", cfg.get("eff_base_mode", "A0"))
        for t in range(len(lambda_t)):
            g = gate_weight[t]
            if eff_anchor == "A0":
                A_eff = A0 + g * (A1 - A0)
            else:
                A_eff = A_base + g * delta_proxy_masked
            A_eff = A_eff * support_mask
            dist_base.append(dist_l1(A_eff, A_base * support_mask, dist_mask))
            dist_reg0.append(dist_l1(A_eff, A0_eff, dist_mask))
            dist_reg1.append(dist_l1(A_eff, A1_eff, dist_mask))
            if pred_edges:
                retained = np.mean([abs(g * delta_proxy_masked[tgt, src]) for (src, tgt) in pred_edges])
            else:
                retained = 0.0
            retained_ratio.append(retained / (raw_strength + eps))

        dist_base = np.array(dist_base, dtype=np.float32)
        dist_reg0 = np.array(dist_reg0, dtype=np.float32)
        dist_reg1 = np.array(dist_reg1, dtype=np.float32)
        retained_ratio = np.array(retained_ratio, dtype=np.float32)
        rel = dist_reg0 - dist_reg1

        if args.sanity and eff_anchor == "A0":
            if high_non_sat is not None and high_non_sat.sum() > 0:
                idx = np.argmax(lambda_t * high_non_sat)
                g = gate_weight[idx]
                A_eff = A0 + g * (A1 - A0)
                A_eff = A_eff * support_mask
                d0 = dist_l1(A_eff, A0_eff, dist_mask)
                print(f"g~0 check: t={idx} g={g:.4f} dist(A_eff,A0)={d0:.6f}")
            # force g=0/1 checks
            A_eff0 = A0 * support_mask
            A_eff1 = A1 * support_mask
            d00 = dist_l1(A_eff0, A0_eff, dist_mask)
            d01 = dist_l1(A_eff0, A1_eff, dist_mask)
            d10 = dist_l1(A_eff1, A0_eff, dist_mask)
            d11 = dist_l1(A_eff1, A1_eff, dist_mask)
            print(f"g_force=0 dist_to_A0={d00:.6f} dist_to_A1={d01:.6f}")
            print(f"g_force=1 dist_to_A0={d10:.6f} dist_to_A1={d11:.6f}")

        # regime-aware metrics
        if t_switch is not None:
            t_idx = np.arange(len(lambda_t))
            pre_mask = valid_mask & (t_idx < t_switch)
            post_mask = valid_mask & (t_idx >= t_switch)
            margin = rel
            align_all_pre = float((rel[pre_mask] > 0).mean()) if pre_mask.sum() > 0 else np.nan
            align_all_post = float((rel[post_mask] > 0).mean()) if post_mask.sum() > 0 else np.nan
            overall_align = float((rel[valid_mask] > 0).mean()) if valid_mask.sum() > 0 else np.nan
            mean_margin_pre = float(margin[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
            mean_margin_post = float(margin[post_mask].mean()) if post_mask.sum() > 0 else np.nan
            rel_pre_mean = float(rel[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
            rel_post_mean = float(rel[post_mask].mean()) if post_mask.sum() > 0 else np.nan
            rel_pre_std = float(rel[pre_mask].std()) if pre_mask.sum() > 0 else np.nan
            rel_post_std = float(rel[post_mask].std()) if post_mask.sum() > 0 else np.nan
            mean_dist_reg0_pre = float(dist_reg0[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
            mean_dist_reg1_pre = float(dist_reg1[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
            mean_dist_reg0_post = float(dist_reg0[post_mask].mean()) if post_mask.sum() > 0 else np.nan
            mean_dist_reg1_post = float(dist_reg1[post_mask].mean()) if post_mask.sum() > 0 else np.nan
        else:
            align_all_pre = align_all_post = overall_align = np.nan
            mean_margin_pre = mean_margin_post = np.nan
            rel_pre_mean = rel_post_mean = np.nan
            rel_pre_std = rel_post_std = np.nan
            mean_dist_reg0_pre = mean_dist_reg1_pre = np.nan
            mean_dist_reg0_post = mean_dist_reg1_post = np.nan
            pre_mask = np.zeros_like(valid_mask, dtype=bool)
            post_mask = np.zeros_like(valid_mask, dtype=bool)

        def stats_pre_post(arr):
            pre = arr[pre_mask]
            post = arr[post_mask]
            def stat(x):
                if x.size == 0:
                    return {"mean": 0.0, "min": 0.0, "max": 0.0}
                return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
            return stat(pre), stat(post)

        lambda_stats_pre, lambda_stats_post = stats_pre_post(lambda_t)
        gate_stats_pre, gate_stats_post = stats_pre_post(gate_weight)

        # subset-conditioned alignment + margin stats
        def margin_stats(mask):
            vals = margin[mask]
            if vals.size == 0:
                return np.nan, np.nan
            return float(vals.mean()), float(vals.std())

        align_low_pre = float((rel[pre_mask & low_mask] > 0).mean()) if (pre_mask & low_mask).sum() > 0 else np.nan
        align_low_post = float((rel[post_mask & low_mask] > 0).mean()) if (post_mask & low_mask).sum() > 0 else np.nan
        if high_non_sat is not None:
            align_high_pre = float((rel[pre_mask & high_non_sat] > 0).mean()) if (pre_mask & high_non_sat).sum() > 0 else np.nan
            align_high_post = float((rel[post_mask & high_non_sat] > 0).mean()) if (post_mask & high_non_sat).sum() > 0 else np.nan
        else:
            align_high_pre = align_high_post = np.nan

        margin_all_pre_mean, margin_all_pre_std = margin_stats(pre_mask)
        margin_all_post_mean, margin_all_post_std = margin_stats(post_mask)
        margin_low_pre_mean, margin_low_pre_std = margin_stats(pre_mask & low_mask)
        margin_low_post_mean, margin_low_post_std = margin_stats(post_mask & low_mask)
        if high_non_sat is not None:
            margin_high_pre_mean, margin_high_pre_std = margin_stats(pre_mask & high_non_sat)
            margin_high_post_mean, margin_high_post_std = margin_stats(post_mask & high_non_sat)
        else:
            margin_high_pre_mean = margin_high_pre_std = np.nan
            margin_high_post_mean = margin_high_post_std = np.nan

        # count stats
        n_pre = int(pre_mask.sum())
        n_post = int(post_mask.sum())
        n_low = int(low_mask.sum())
        n_low_pre = int((pre_mask & low_mask).sum())
        n_low_post = int((post_mask & low_mask).sum())
        n_high_ns_pre = int((pre_mask & high_non_sat).sum()) if high_non_sat is not None else 0
        n_high_ns_post = int((post_mask & high_non_sat).sum()) if high_non_sat is not None else 0
        low_post_min = int(cfg.get("low_post_min", 10))
        if n_low_post < low_post_min:
            logs.append("WARN: low_post too small")
            align_low_post = np.nan
            margin_low_post_mean = np.nan
            margin_low_post_std = np.nan

        if pred_edges:
            mean_abs_g_delta_t = np.zeros(len(lambda_t), dtype=np.float32)
            for t in range(len(lambda_t)):
                mean_abs_g_delta_t[t] = float(np.mean([abs(gate_weight[t] * delta_proxy_masked[tgt, src])
                                                       for (src, tgt) in pred_edges]))
        else:
            mean_abs_g_delta_t = np.zeros(len(lambda_t), dtype=np.float32)

        def subset_row(name, mask):
            mean_abs_g_delta = mean_over_mask(mean_abs_g_delta_t, mask)
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
                "mean_abs_g_delta": mean_abs_g_delta,
                "delta_mag_mean": delta_mag_mean,
                "delta_mag_max": delta_mag_max,
                "align_all_pre": align_all_pre,
                "align_all_post": align_all_post,
                "align_low_pre": align_low_pre,
                "align_low_post": align_low_post,
                "align_highns_pre": align_high_pre,
                "align_highns_post": align_high_post,
                "overall_align": overall_align,
                "mean_margin_pre": mean_margin_pre,
                "mean_margin_post": mean_margin_post,
                "rel_pre_mean": rel_pre_mean,
                "rel_post_mean": rel_post_mean,
                "rel_pre_std": rel_pre_std,
                "rel_post_std": rel_post_std,
                "margin_all_pre_mean": margin_all_pre_mean,
                "margin_all_pre_std": margin_all_pre_std,
                "margin_all_post_mean": margin_all_post_mean,
                "margin_all_post_std": margin_all_post_std,
                "margin_low_pre_mean": margin_low_pre_mean,
                "margin_low_pre_std": margin_low_pre_std,
                "margin_low_post_mean": margin_low_post_mean,
                "margin_low_post_std": margin_low_post_std,
                "margin_high_pre_mean": margin_high_pre_mean,
                "margin_high_pre_std": margin_high_pre_std,
                "margin_high_post_mean": margin_high_post_mean,
                "margin_high_post_std": margin_high_post_std,
                "n_pre": n_pre,
                "n_post": n_post,
                "n_low": n_low,
                "n_low_pre": n_low_pre,
                "n_low_post": n_low_post,
                "n_high_ns_pre": n_high_ns_pre,
                "n_high_ns_post": n_high_ns_post,
            }

        summary_rows = [
            subset_row("high_non_sat", high_non_sat) if high_non_sat is not None else None,
            subset_row("low", low_mask),
            subset_row("all", all_mask),
        ]
        summary_rows = [r for r in summary_rows if r is not None]

        if cfg.get("output_high_sat", False):
            summary_rows.append(subset_row("high_sat", high_sat))

        summary_csv = os.path.join(out_dir, f"step5pp_summary{tag}.csv")
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write(",".join(summary_rows[0].keys()) + "\n")
            for r in summary_rows:
                f.write(",".join([str(r[k]) for k in summary_rows[0].keys()]) + "\n")

        summary_md = os.path.join(out_dir, f"step5pp_summary{tag}.md")
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("## Step5++ Summary\n\n")
            f.write("| " + " | ".join(summary_rows[0].keys()) + " |\n")
            f.write("| " + " | ".join(["---"] * len(summary_rows[0])) + " |\n")
            for r in summary_rows:
                f.write("| " + " | ".join([str(r[k]) for k in summary_rows[0].keys()]) + " |\n")
            f.write("\n")
            f.write("- high_non_sat should have smallest dist_base\n")
            f.write("- low should have larger dist_base and higher retained_ratio\n")

        retained_csv = os.path.join(out_dir, f"retained_summary{tag}.csv")
        with open(retained_csv, "w", encoding="utf-8") as f:
            f.write("t,retained_ratio\n")
            for i, v in enumerate(retained_ratio):
                f.write(f"{i},{v:.6f}\n")

        if write_plots and HAS_MPL:
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
            # overlay subsets
            if high_non_sat is not None:
                axes[0].fill_between(np.arange(len(lambda_t)), 0, 1, where=high_non_sat, color="red", alpha=0.1)
            axes[0].fill_between(np.arange(len(lambda_t)), 0, 1, where=low_mask, color="green", alpha=0.1)
            axes[0].legend(loc="upper right")
            axes[0].set_title("lambda(t) and gate_weight(t)")

            axes[1].plot(dist_base, label="dist_base")
            axes[1].plot(dist_reg0, label="dist_reg0")
            axes[1].plot(dist_reg1, label="dist_reg1")
            axes[1].legend(loc="upper right")
            axes[1].set_title("distance curves")

            axes[2].plot(retained_ratio, color="tab:purple", label="retained_ratio")
            axes[2].plot(rel, color="tab:green", label="rel=reg0-reg1")
            axes[2].legend(loc="upper right")
            axes[2].set_title("retained_ratio(t) and rel(t)")
            axes[2].set_xlabel("t")
            fig.tight_layout()
            fig.savefig(sim_fig, dpi=200)
            plt.close(fig)

        mean_retained_high = None
        if high_non_sat is not None:
            mean_retained_high = mean_over_mask(retained_ratio, high_non_sat)
        elif high_sat is not None:
            mean_retained_high = mean_over_mask(retained_ratio, high_sat)
        else:
            mean_retained_high = 0.0
        mean_retained_low = mean_over_mask(retained_ratio, low_mask)
        # sanity prints
        if args.sanity:
            def subset_print(name, mask):
                ml = mean_over_mask(lambda_t, mask)
                mg = mean_over_mask(gate_weight, mask)
                print(f"{name}: count={int(mask.sum())} mean_lambda={ml:.6f} mean_gate_weight={mg:.6f}")
            print(f"subset_strategy={subset_strategy} high_thr={high_thr:.6f} low_thr={low_thr:.6f}")
            if high_non_sat is not None:
                subset_print("high_non_sat", high_non_sat)
            if cfg.get("output_high_sat", False) and high_sat is not None:
                subset_print("high_sat", high_sat)
            subset_print("low", low_mask)
            subset_print("all", all_mask)
            vb = dist_base[valid_mask]
            v0 = dist_reg0[valid_mask]
            v1 = dist_reg1[valid_mask]
            print(f"dist_std: base={vb.std():.6f} reg0={v0.std():.6f} reg1={v1.std():.6f}")
            print(f"align: pre={align_all_pre:.6f} post={align_all_post:.6f} overall={overall_align:.6f}")
            print(f"margin: pre={mean_margin_pre:.6f} post={mean_margin_post:.6f}")
            print(f"rel: pre_mean={rel_pre_mean:.6f} pre_std={rel_pre_std:.6f} post_mean={rel_post_mean:.6f} post_std={rel_post_std:.6f}")
            print(f"mask: delta_mask_mode={delta_mask_mode} dist_mask_mode={dist_mask_mode}")
            print(f"mean(dist_reg0_pre)={mean_dist_reg0_pre:.6f} mean(dist_reg1_pre)={mean_dist_reg1_pre:.6f}")
            print(f"mean(dist_reg0_post)={mean_dist_reg0_post:.6f} mean(dist_reg1_post)={mean_dist_reg1_post:.6f}")
            if mean_dist_reg0_pre >= mean_dist_reg1_pre or mean_dist_reg1_post >= mean_dist_reg0_post:
                print("WARN: A0/A1 may be swapped wrt t_switch.")
            if not np.isnan(align_all_pre) and pre_mask.sum() > 0:
                frac = float((rel[pre_mask] > 0).mean())
                if abs(align_all_pre - frac) > 1e-6:
                    print("WARN: align_pre != fraction(rel_pre>0)")
            # directional sanity checks
            mean_gate_high = mean_over_mask(gate_weight, high_non_sat) if high_non_sat is not None else np.nan
            mean_gate_low = mean_over_mask(gate_weight, low_mask)
            if not np.isnan(mean_gate_high) and not np.isnan(mean_gate_low):
                if mean_gate_high < mean_gate_low:
                    print("[OK] gate direction")
                else:
                    print("WARN: gate direction mismatch")
            if high_non_sat is not None:
                if mean_over_mask(dist_reg0, high_non_sat) < mean_over_mask(dist_reg1, high_non_sat):
                    print("[OK] high subset closer to A0")
                else:
                    print("WARN: high subset not closer to A0")
            if mean_over_mask(dist_reg1, low_mask) < mean_over_mask(dist_reg0, low_mask):
                print("[OK] low subset closer to A1")
            else:
                print("WARN: low subset not closer to A1")

        if write_plots and not HAS_MPL:
            logs.append("WARN: matplotlib not available; skipping plots.")

        return summary_rows, {
            "subset_strategy": subset_strategy,
            "high_thr": None,
            "low_thr": low_thr,
            "delta_mag_mean": delta_mag_mean,
            "delta_mag_max": delta_mag_max,
            "overall_align": overall_align,
            "mean_retained_ratio_high": mean_retained_high,
            "mean_retained_ratio_low": mean_retained_low,
            "dist_std_base": float(dist_base[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "dist_std_reg0": float(dist_reg0[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "dist_std_reg1": float(dist_reg1[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "align_all_pre": align_all_pre,
            "align_all_post": align_all_post,
            "align_low_pre": align_low_pre,
            "align_low_post": align_low_post,
            "align_high_pre": align_high_pre,
            "align_high_post": align_high_post,
            "mean_margin_pre": mean_margin_pre,
            "mean_margin_post": mean_margin_post,
            "rel_pre_mean": rel_pre_mean,
            "rel_post_mean": rel_post_mean,
            "corr_lambda_dist_base": corrcoef_safe(lambda_t, dist_base),
            "corr_gate_dist_base": corrcoef_safe(gate_weight, dist_base),
            "corr_lambda_retained": corrcoef_safe(lambda_t, retained_ratio),
            "lambda_stats_pre": lambda_stats_pre,
            "lambda_stats_post": lambda_stats_post,
            "gate_stats_pre": gate_stats_pre,
            "gate_stats_post": gate_stats_post,
            "delta_mask_mode": delta_mask_mode,
            "dist_mask_mode": dist_mask_mode,
            "dist_mask_nnz": int((dist_mask != 0).sum()),
            "n_pre": n_pre,
            "n_post": n_post,
            "n_low": n_low,
            "n_low_pre": n_low_pre,
            "n_low_post": n_low_post,
            "n_high_ns_pre": n_high_ns_pre,
            "n_high_ns_post": n_high_ns_post,
        }

    # lambda handling (single vs top_m)
    if args.score_type:
        x_path = os.path.join(data_dir, "X.npy")
        X = np.load(x_path)
        configs = pick_lambda_configs_from_step4(data_dir, args.score_type, args.top_m)
        compare_rows = []
        for i, c in enumerate(configs, start=1):
            lambda_t, valid_mask = compute_lambda_kmeans(X, c["window"], c["k"])
            tag = f"_{c['window']}_{c['k']}"
            summary_rows, metrics = run_once(lambda_t, valid_mask, tag, write_plots=(i == 1))
            compare_rows.append({
                "config_id": tag,
                "window": c["window"],
                "k": c["k"],
                "score": c["score"],
                "overall_align": metrics["overall_align"],
                "retained_gap": metrics["mean_retained_ratio_low"] - metrics["mean_retained_ratio_high"],
            })
        compare_csv = os.path.join(out_dir, "step5pp_compare.csv")
        with open(compare_csv, "w", encoding="utf-8") as f:
            f.write("config_id,window,k,score,overall_align,retained_gap\n")
            for r in compare_rows:
                f.write(f"{r['config_id']},{r['window']},{r['k']},{r['score']},{r['overall_align']},{r['retained_gap']}\n")
        config_used = {
            "data_dir": data_dir,
            "config_path": cfg_path,
            "pred_prefix": cfg.get("pred_prefix", ""),
            "score_type": args.score_type,
            "top_m": args.top_m,
            "topk_mode": topk_mode,
            "top_k_source": k_source,
            "edge_mask": cfg.get("edge_mask", "base_only"),
            "delta_mask_mode": cfg.get("delta_mask", "union_base_predchange"),
            "dist_mask_mode": cfg.get("dist_mask", "union_base_predchange"),
            "regime_support_mode": cfg.get("regime_support_mode", "union_base_predchange"),
            "eff_anchor": cfg.get("eff_anchor", cfg.get("eff_base_mode", "A0")),
            "gate_fn": "soft: g=1-lambda" if gate_mode == "soft" else "hard: g=1(lambda<thr)",
            "low_post_min": int(cfg.get("low_post_min", 10)),
        }
        with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(config_used, f, indent=2)
    else:
        lambda_t, valid_mask, lambda_source, t_switch = load_lambda_and_mask(data_dir, logs)
        if valid_mask.sum() > 0:
            v = lambda_t[valid_mask]
            v = v[np.isfinite(v)]
            if v.size > 0:
                logs.append(f"lambda stats: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
        summary_rows, metrics = run_once(lambda_t, valid_mask, tag="", write_plots=True)
        if args.sanity:
            # diagnostic run: focus on true change edges with A0 base
            saved_dist_mask = cfg.get("dist_mask", "union_base_predchange")
            saved_eff_anchor = cfg.get("eff_anchor", cfg.get("eff_base_mode", "A0"))
            cfg["dist_mask"] = "true_change_only"
            cfg["eff_anchor"] = "A0"
            _, diag = run_once(lambda_t, valid_mask, tag="_diag", write_plots=False)
            print(f"diagnostic rel: pre_mean={diag.get('rel_pre_mean')} post_mean={diag.get('rel_post_mean')}")
            cfg["dist_mask"] = saved_dist_mask
            cfg["eff_anchor"] = saved_eff_anchor
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
            "gate_fn": "soft: g=1-lambda" if gate_mode == "soft" else "hard: g=1(lambda<thr)",
            "tau_hard": tau_hard,
            "w_soft": w_soft,
            "subset_high_q": cfg.get("subset_high_q", 0.90),
            "subset_low_q": cfg.get("subset_low_q", 0.50),
            "edge_mask": cfg.get("edge_mask", "base_only"),
            "delta_mask_mode": metrics.get("delta_mask_mode"),
            "dist_mask_mode": metrics.get("dist_mask_mode"),
            "regime_support_mode": cfg.get("regime_support_mode", "union_base_predchange"),
            "eff_anchor": cfg.get("eff_anchor", cfg.get("eff_base_mode", "A0")),
            "output_topk_edges": top_k,
            "topk_mode": topk_mode,
            "top_k_source": k_source,
            "K_true": K_true,
            "K_pred": len(pred_edges),
            "lambda_stats_pre": metrics.get("lambda_stats_pre"),
            "lambda_stats_post": metrics.get("lambda_stats_post"),
            "gate_stats_pre": metrics.get("gate_stats_pre"),
            "gate_stats_post": metrics.get("gate_stats_post"),
            "dist_std_base": metrics.get("dist_std_base"),
            "dist_std_reg0": metrics.get("dist_std_reg0"),
            "dist_std_reg1": metrics.get("dist_std_reg1"),
            "align_low_post": metrics.get("align_low_post"),
            "dist_mask_nnz": metrics.get("dist_mask_nnz"),
            "low_post_min": int(cfg.get("low_post_min", 10)),
            "n_pre": metrics.get("n_pre"),
            "n_post": metrics.get("n_post"),
            "n_low": metrics.get("n_low"),
            "n_low_pre": metrics.get("n_low_pre"),
            "n_low_post": metrics.get("n_low_post"),
            "n_high_ns_pre": metrics.get("n_high_ns_pre"),
            "n_high_ns_post": metrics.get("n_high_ns_post"),
        }
        with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(config_used, f, indent=2)

    # Sanity header in logs
    header = [
        "=== Sanity Header ===",
        f"data_dir={data_dir}",
        f"config_path={cfg_path}",
        f"gate_fn={'soft: g=1-lambda' if gate_mode == 'soft' else 'hard: g=1(lambda<thr)'}",
    ]
    if not args.score_type:
        header.extend([
            "rel = dist_reg0 - dist_reg1; rel>0 => closer to reg1",
            f"delta_mask_mode={config_used.get('delta_mask_mode')}",
            f"dist_mask_mode={config_used.get('dist_mask_mode')}",
            f"regime_support_mode={config_used.get('regime_support_mode')}",
            f"eff_anchor={config_used.get('eff_anchor')}",
            f"topk_mode={config_used.get('topk_mode')}",
            f"k_source={config_used.get('top_k_source')}",
            f"lambda_stats_pre={config_used.get('lambda_stats_pre')}",
            f"lambda_stats_post={config_used.get('lambda_stats_post')}",
            f"gate_stats_pre={config_used.get('gate_stats_pre')}",
            f"gate_stats_post={config_used.get('gate_stats_post')}",
        ])
    logs = header + logs
    write_logs(logs, os.path.join(out_dir, "logs.txt"))

    if args.sanity:
        print("=== Step5++ sanity ===")
        print(f"A_base nnz={int((np.abs(A_base) > 0).sum())}")
        print(f"A0 min/max/mean: {A0.min():.4f}/{A0.max():.4f}/{A0.mean():.4f}")
        print(f"A1 min/max/mean: {A1.min():.4f}/{A1.max():.4f}/{A1.mean():.4f}")
        print(f"K_true={K_true}, K_pred={len(pred_edges)} (source={k_source})")


if __name__ == "__main__":
    main()
