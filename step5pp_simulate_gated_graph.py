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
    raw_cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

    defaults = {
        "pred_prefix": "cmiknn",
        "score_type": "valdiff",
        "delta_mode": "A1_minus_A0",
        "gate_mode": "soft",
        "tau_hard": 0.8,
        "w_soft": None,
        "subset_high_q": 0.90,
        "subset_low_q": 0.50,
        "delta_mask_mode": "union_base_predchange",
        "dist_mask_mode": "true_change_only",
        "regime_support_mode": "union_base_predchange",
        "regime_ref_source": "estimated",
        "auto_swap_regimes": False,
        "swap_decision_by": "pre_rel_mean",
        "norm": "none",
        "output_topk_edges": 20,
        "topk_mode": "match_true",
        "top_k": 20,
        "output_high_sat": False,
        "low_post_min": 10,
        "switch_band_pre": 300,
        "switch_band_post": 300,
        "switch_window": 200,
        "switch_window_sweep": [],
        "directional_align_overall_min": 0.55,
        "switch_band_correct_rate_min": 0.55,
        "switch_pre_correct_rate_min": 0.50,
        "switch_post_correct_rate_min": 0.50,
        "auc_switch_rel_min": 0.55,
        "directional_align_overall_min_v3": 0.60,
        "switch_band_correct_rate_min_v3": 0.60,
        "switch_margin_min_v3": 0.0,
        "peak_delay_max_frac_v3": 0.5,
        "retained_gap_switch_min_v3": 0.10,
    }

    cfg = defaults.copy()
    cfg.update(raw_cfg)
    compat_warnings = []

    # Backward compatibility for legacy mask fields.
    if "delta_mask_mode" not in raw_cfg:
        if "delta_mask" in raw_cfg:
            cfg["delta_mask_mode"] = raw_cfg["delta_mask"]
            compat_warnings.append("legacy field `delta_mask` mapped to `delta_mask_mode`")
        elif "edge_mask" in raw_cfg:
            cfg["delta_mask_mode"] = raw_cfg["edge_mask"]
            compat_warnings.append("legacy field `edge_mask` mapped to `delta_mask_mode`")
    if "dist_mask_mode" not in raw_cfg:
        if "dist_mask" in raw_cfg:
            cfg["dist_mask_mode"] = raw_cfg["dist_mask"]
            compat_warnings.append("legacy field `dist_mask` mapped to `dist_mask_mode`")
        elif "edge_mask" in raw_cfg:
            cfg["dist_mask_mode"] = raw_cfg["edge_mask"]
            compat_warnings.append("legacy field `edge_mask` mapped to `dist_mask_mode`")

    # Keep compatibility with old short alias.
    if cfg.get("regime_ref_source") == "gt":
        cfg["regime_ref_source"] = "ground_truth"
    cfg["_compat_warnings"] = compat_warnings
    return cfg


def validate_cfg(cfg):
    valid_delta_modes = {"base_only", "union_base_predchange"}
    valid_dist_modes = {"base_only", "union_base_predchange", "true_change_only", "union_delta_topk"}
    valid_gate_modes = {"soft", "hard"}

    delta_mask_mode = cfg.get("delta_mask_mode")
    dist_mask_mode = cfg.get("dist_mask_mode")
    gate_mode = cfg.get("gate_mode")

    if delta_mask_mode not in valid_delta_modes:
        raise ValueError(f"Invalid delta_mask_mode={delta_mask_mode}, expected one of {sorted(valid_delta_modes)}")
    if dist_mask_mode not in valid_dist_modes:
        raise ValueError(f"Invalid dist_mask_mode={dist_mask_mode}, expected one of {sorted(valid_dist_modes)}")
    if gate_mode not in valid_gate_modes:
        raise ValueError(f"Invalid gate_mode={gate_mode}, expected one of {sorted(valid_gate_modes)}")


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


def auc_binary_safe(y_true, scores):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(scores)
    y_true = y_true[mask]
    scores = scores[mask]
    if y_true.size == 0:
        return np.nan
    y = (y_true > 0.5).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[y == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def peak_delay_switch(signal, valid_mask, t_switch, switch_window):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if signal.size < 3:
        return np.nan
    t_switch = int(t_switch)
    w = max(1, int(switch_window))
    lo = max(1, t_switch - w)
    hi = min(signal.size - 1, t_switch + w)
    if hi <= lo:
        return np.nan
    idx = np.arange(lo, hi, dtype=int)
    step = np.abs(signal[idx] - signal[idx - 1])
    step_valid = valid_mask[idx] & valid_mask[idx - 1] & np.isfinite(step)
    if int(step_valid.sum()) == 0:
        return np.nan
    score = np.where(step_valid, step, -np.inf)
    peak_t = int(idx[int(np.argmax(score))])
    return float(abs(peak_t - t_switch))


def corr_time_in_mask(values, mask):
    values = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if int(mask.sum()) < 2:
        return np.nan
    t = np.arange(values.shape[0], dtype=float)[mask]
    v = values[mask]
    if int(np.isfinite(v).sum()) < 2:
        return np.nan
    return corrcoef_safe(t, v)


def csv_safe(v):
    if v is None:
        return "NaN"
    if isinstance(v, (bool, np.bool_)):
        return "True" if bool(v) else "False"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "NaN"
        return str(float(v))
    return str(v)


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


def _delta_topk_mask(delta_ref, union_mask, topk=6, thr=None):
    delta_ref = np.abs(delta_ref)
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


def build_delta_mask(mode, base_mask, pred_mask, true_mask, delta_ref, cfg):
    # Fixed definitions:
    # - base_only: A_base != 0
    # - union_base_predchange: (A_base != 0) | (pred_adj != 0)
    if mode in ("base_only", "union_base_predchange"):
        mask = build_mask(mode, base_mask, pred_mask)
    else:
        raise ValueError(f"Unsupported delta_mask_mode={mode}")
    np.fill_diagonal(mask, 0.0)
    return mask.astype(np.float32)


def build_dist_mask(mode, base_mask, pred_mask, true_mask, delta_ref, cfg, pred_edges):
    # Fixed definitions:
    # - base_only: A_base != 0
    # - union_base_predchange: (A_base != 0) | (pred_adj != 0)
    # - true_change_only: adj_true != 0
    # - union_delta_topk: pred_adj != 0
    if mode in ("base_only", "union_base_predchange"):
        mask = build_mask(mode, base_mask, pred_mask)
    elif mode == "true_change_only":
        mask = (true_mask != 0).astype(np.float32)
    elif mode == "union_delta_topk":
        mask = (pred_mask != 0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dist_mask_mode={mode}")
    np.fill_diagonal(mask, 0.0)
    return mask.astype(np.float32)


def _mean_offdiag_abs(x):
    if x.ndim != 2:
        return float(np.mean(np.abs(x)))
    mask = np.ones_like(x, dtype=bool)
    np.fill_diagonal(mask, False)
    vals = np.abs(x[mask])
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def maybe_swap_regimes(A0, A1, t_switch, lambda_t, valid_mask, cfg, gate_mode, tau_hard, logs, diagnostic_mask):
    auto_swap = bool(cfg.get("auto_swap_regimes", False))
    decision_by = cfg.get("swap_decision_by", "pre_rel_mean")
    if not auto_swap and t_switch is None:
        return A0, A1, False, "auto_swap_regimes_disabled+t_switch_missing"
    if not auto_swap:
        logs.append("WARN: auto_swap_regimes=False, will only diagnose without swapping.")
    if t_switch is None:
        return A0, A1, False, "t_switch_missing"
    if lambda_t is None or valid_mask is None:
        return A0, A1, False, "lambda_or_valid_mask_missing"

    t_idx = np.arange(len(lambda_t))
    pre_mask = valid_mask & (t_idx < int(t_switch))
    post_mask = valid_mask & (t_idx >= int(t_switch))
    if pre_mask.sum() == 0:
        return A0, A1, False, "pre_segment_empty"

    if gate_mode == "soft":
        gate_weight = np.clip(1.0 - lambda_t, 0.0, 1.0)
    else:
        gate_weight = (lambda_t < tau_hard).astype(np.float32)
    gate_weight = np.where(valid_mask, gate_weight, 0.0)

    rel = np.zeros(len(lambda_t), dtype=np.float32)
    for t in range(len(lambda_t)):
        if not valid_mask[t]:
            continue
        g = gate_weight[t]
        A_eff = A0 + g * (A1 - A0)
        d0 = dist_l1(A_eff, A0, diagnostic_mask)
        d1 = dist_l1(A_eff, A1, diagnostic_mask)
        rel[t] = d0 - d1

    pre_rel_mean = float(rel[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
    post_rel_mean = float(rel[post_mask].mean()) if post_mask.sum() > 0 else np.nan

    if decision_by == "pre_rel_mean":
        should_swap = bool(pre_rel_mean > 0)
    elif decision_by == "pre_post_rel":
        post_ok = True if post_mask.sum() == 0 else bool(post_rel_mean < 0)
        should_swap = bool(pre_rel_mean > 0 and post_ok)
    else:
        raise ValueError(f"Unknown swap_decision_by: {decision_by}")

    reason = (
        f"{decision_by}: pre_rel_mean={pre_rel_mean:.6f}, "
        f"post_rel_mean={post_rel_mean:.6f}, pre_count={int(pre_mask.sum())}, "
        f"post_count={int(post_mask.sum())}"
    )
    if should_swap and auto_swap:
        logs.append(f"A0/A1 auto-swapped. reason={reason}")
        return A1, A0, True, reason
    if should_swap and not auto_swap:
        logs.append(f"WARN: A0/A1 looks swapped but auto_swap_regimes=False. reason={reason}")
        return A0, A1, False, reason
    logs.append(f"A0/A1 kept. reason={reason}")
    return A0, A1, False, reason


def evaluate_gate_direction_checks(rel, pre_mask, post_mask, low_mask, high_non_sat, gate_weight, dist_reg0, dist_reg1):
    margin = rel
    # rel = dist_reg0 - dist_reg1; rel>0 means closer to A1, rel<0 means closer to A0.
    # Expected sign: pre closer to A0 -> rel<0; post closer to A1 -> rel>0.
    align_pre = float((rel[pre_mask] < 0).mean()) if pre_mask.sum() > 0 else np.nan
    align_post = float((rel[post_mask] > 0).mean()) if post_mask.sum() > 0 else np.nan
    n_all = int(pre_mask.sum() + post_mask.sum())
    if n_all > 0:
        align_overall = float(
            ((rel[pre_mask] < 0).sum() + (rel[post_mask] > 0).sum()) / max(n_all, 1)
        )
    else:
        align_overall = np.nan
    mean_margin_pre = float(margin[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
    mean_margin_post = float(margin[post_mask].mean()) if post_mask.sum() > 0 else np.nan
    rel_pre_mean = float(rel[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
    rel_post_mean = float(rel[post_mask].mean()) if post_mask.sum() > 0 else np.nan
    rel_pre_std = float(rel[pre_mask].std()) if pre_mask.sum() > 0 else np.nan
    rel_post_std = float(rel[post_mask].std()) if post_mask.sum() > 0 else np.nan

    align_low_pre = float((rel[pre_mask & low_mask] < 0).mean()) if (pre_mask & low_mask).sum() > 0 else np.nan
    align_low_post = float((rel[post_mask & low_mask] > 0).mean()) if (post_mask & low_mask).sum() > 0 else np.nan
    if high_non_sat is not None:
        align_high_pre = float((rel[pre_mask & high_non_sat] < 0).mean()) if (pre_mask & high_non_sat).sum() > 0 else np.nan
        align_high_post = float((rel[post_mask & high_non_sat] > 0).mean()) if (post_mask & high_non_sat).sum() > 0 else np.nan
    else:
        align_high_pre = np.nan
        align_high_post = np.nan

    mean_gate_high = float(gate_weight[high_non_sat].mean()) if (high_non_sat is not None and high_non_sat.sum() > 0) else np.nan
    mean_gate_low = float(gate_weight[low_mask].mean()) if low_mask.sum() > 0 else np.nan
    mean_dist_reg0_pre = float(dist_reg0[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
    mean_dist_reg1_pre = float(dist_reg1[pre_mask].mean()) if pre_mask.sum() > 0 else np.nan
    mean_dist_reg0_post = float(dist_reg0[post_mask].mean()) if post_mask.sum() > 0 else np.nan
    mean_dist_reg1_post = float(dist_reg1[post_mask].mean()) if post_mask.sum() > 0 else np.nan

    check_gate_direction = (
        bool(mean_gate_high < mean_gate_low)
        if (np.isfinite(mean_gate_high) and np.isfinite(mean_gate_low))
        else None
    )
    high_rel_mean = float(rel[high_non_sat].mean()) if (high_non_sat is not None and high_non_sat.sum() > 0) else np.nan
    low_rel_mean = float(rel[low_mask].mean()) if low_mask.sum() > 0 else np.nan
    check_high_closer_a0 = bool(high_rel_mean < 0) if np.isfinite(high_rel_mean) else None
    check_low_closer_a1 = bool(low_rel_mean > 0) if np.isfinite(low_rel_mean) else None
    pre_rel_sign_ok = bool(rel_pre_mean < 0) if np.isfinite(rel_pre_mean) else None
    post_rel_sign_ok = bool(rel_post_mean > 0) if np.isfinite(rel_post_mean) else None

    checks = [v for v in [pre_rel_sign_ok, post_rel_sign_ok] if v is not None]
    check_overall_pass = bool(all(checks)) if checks else False

    return {
        "align_pre": align_pre,
        "align_post": align_post,
        "align_overall": align_overall,
        "align_all_pre": align_pre,
        "align_all_post": align_post,
        "overall_align": align_overall,
        "mean_margin_pre": mean_margin_pre,
        "mean_margin_post": mean_margin_post,
        "rel_pre_mean": rel_pre_mean,
        "rel_post_mean": rel_post_mean,
        "rel_pre_std": rel_pre_std,
        "rel_post_std": rel_post_std,
        "align_low_pre": align_low_pre,
        "align_low_post": align_low_post,
        "align_high_pre": align_high_pre,
        "align_high_post": align_high_post,
        "mean_gate_high": mean_gate_high,
        "mean_gate_low": mean_gate_low,
        "mean_dist_reg0_pre": mean_dist_reg0_pre,
        "mean_dist_reg1_pre": mean_dist_reg1_pre,
        "mean_dist_reg0_post": mean_dist_reg0_post,
        "mean_dist_reg1_post": mean_dist_reg1_post,
        "check_gate_direction": check_gate_direction,
        "check_high_closer_a0": check_high_closer_a0,
        "check_low_closer_a1": check_low_closer_a1,
        "pre_rel_sign_ok": pre_rel_sign_ok,
        "post_rel_sign_ok": post_rel_sign_ok,
        "check_overall_pass": check_overall_pass,
    }


def evaluate_switch_aware_metrics(rel, lambda_t, gate_weight, retained_ratio, valid_mask, t_switch, cfg):
    out = {
        "pre_correct_rate": np.nan,
        "post_correct_rate": np.nan,
        "directional_align_pre": np.nan,
        "directional_align_post": np.nan,
        "directional_align_overall": np.nan,
        "switch_window": int(cfg.get("switch_window", 200)),
        "switch_pre_correct_rate": np.nan,
        "switch_post_correct_rate": np.nan,
        "switch_band_correct_rate": np.nan,
        "switch_margin_pre": np.nan,
        "switch_margin_post": np.nan,
        "corr_lambda_regime": np.nan,
        "corr_gate_regime": np.nan,
        "corr_retained_regime": np.nan,
        "auc_switch_lambda": np.nan,
        "auc_switch_gate": np.nan,
        "auc_switch_rel": np.nan,
        "retained_gap_switch": np.nan,
        "peak_delay_lambda": np.nan,
        "peak_delay_gate": np.nan,
        "peak_delay_rel": np.nan,
        "corr_time_lambda_switch": np.nan,
        "corr_time_gate_switch": np.nan,
        "corr_time_retained_switch": np.nan,
        "switch_band_pass": False,          # v3
        "directional_align_pass": False,    # v3
        "switch_margin_pass": False,        # v3
        "peak_delay_pass": False,           # v3
        "retained_gap_switch_pass": False,  # v3
        "directional_align_pass_v2": False,
        "switch_band_pass_v2": False,
        "switch_pre_pass_v2": False,
        "switch_post_pass_v2": False,
        "switch_auc_pass": False,
        "switch_nan_reasons": [],
    }
    reasons = []
    if t_switch is None:
        reasons.append("t_switch_missing")
        out["switch_nan_reasons"] = reasons
        return out

    t_idx = np.arange(len(lambda_t))
    pre_mask = valid_mask & (t_idx < int(t_switch))
    post_mask = valid_mask & (t_idx >= int(t_switch))
    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())
    if n_pre == 0 or n_post == 0:
        reasons.append("pre_or_post_empty")
        out["switch_nan_reasons"] = reasons
        return out

    correct = np.where(pre_mask, rel < 0, np.where(post_mask, rel > 0, False))
    pre_correct_rate = float((rel[pre_mask] < 0).mean()) if n_pre > 0 else np.nan
    post_correct_rate = float((rel[post_mask] > 0).mean()) if n_post > 0 else np.nan
    directional_align_overall = float(correct[pre_mask | post_mask].mean()) if (n_pre + n_post) > 0 else np.nan

    if cfg.get("switch_window") is not None:
        switch_window = max(1, int(cfg.get("switch_window", 200)))
    else:
        band_pre_compat = max(0, int(cfg.get("switch_band_pre", 300)))
        band_post_compat = max(0, int(cfg.get("switch_band_post", 300)))
        switch_window = max(1, band_pre_compat, band_post_compat)
    band_pre = switch_window
    band_post = switch_window
    lo = max(0, int(t_switch) - max(band_pre, 0))
    hi = min(len(lambda_t), int(t_switch) + max(band_post, 0))
    switch_pre_mask = valid_mask & (t_idx >= lo) & (t_idx < int(t_switch))
    switch_post_mask = valid_mask & (t_idx >= int(t_switch)) & (t_idx < hi)
    switch_band_mask = switch_pre_mask | switch_post_mask
    if int(switch_pre_mask.sum()) == 0:
        reasons.append("switch_pre_empty")
    if int(switch_post_mask.sum()) == 0:
        reasons.append("switch_post_empty")
    if int(switch_band_mask.sum()) == 0:
        reasons.append("switch_band_empty")

    switch_pre_correct_rate = float((rel[switch_pre_mask] < 0).mean()) if switch_pre_mask.sum() > 0 else np.nan
    switch_post_correct_rate = float((rel[switch_post_mask] > 0).mean()) if switch_post_mask.sum() > 0 else np.nan
    switch_band_correct_rate = float(correct[switch_band_mask].mean()) if switch_band_mask.sum() > 0 else np.nan
    # Margin is defined as "correct direction margin" (positive is better for both sides).
    switch_margin_pre = float((-rel[switch_pre_mask]).mean()) if switch_pre_mask.sum() > 0 else np.nan
    switch_margin_post = float(rel[switch_post_mask].mean()) if switch_post_mask.sum() > 0 else np.nan

    regime = np.full(len(lambda_t), np.nan, dtype=np.float64)
    regime[pre_mask] = 0.0
    regime[post_mask] = 1.0
    corr_lambda_regime = corrcoef_safe(lambda_t[valid_mask], regime[valid_mask])
    corr_gate_regime = corrcoef_safe(gate_weight[valid_mask], regime[valid_mask])
    corr_retained_regime = corrcoef_safe(retained_ratio[valid_mask], regime[valid_mask])

    # Positive orientation: post=1 should score higher.
    auc_switch_lambda = auc_binary_safe(regime[valid_mask], -lambda_t[valid_mask])
    auc_switch_gate = auc_binary_safe(regime[valid_mask], gate_weight[valid_mask])
    auc_switch_rel = auc_binary_safe(regime[valid_mask], rel[valid_mask])

    retained_pre = float(retained_ratio[switch_pre_mask].mean()) if switch_pre_mask.sum() > 0 else np.nan
    retained_post = float(retained_ratio[switch_post_mask].mean()) if switch_post_mask.sum() > 0 else np.nan
    retained_gap_switch = retained_pre - retained_post if (np.isfinite(retained_pre) and np.isfinite(retained_post)) else np.nan

    peak_delay_lambda = peak_delay_switch(lambda_t, valid_mask, int(t_switch), switch_window)
    peak_delay_gate = peak_delay_switch(gate_weight, valid_mask, int(t_switch), switch_window)
    peak_delay_rel = peak_delay_switch(rel, valid_mask, int(t_switch), switch_window)
    corr_time_lambda_switch = corr_time_in_mask(lambda_t, switch_band_mask)
    corr_time_gate_switch = corr_time_in_mask(gate_weight, switch_band_mask)
    corr_time_retained_switch = corr_time_in_mask(retained_ratio, switch_band_mask)
    if not np.isfinite(peak_delay_lambda):
        reasons.append("peak_delay_lambda_nan")
    if not np.isfinite(peak_delay_gate):
        reasons.append("peak_delay_gate_nan")
    if not np.isfinite(peak_delay_rel):
        reasons.append("peak_delay_rel_nan")

    directional_align_min = float(cfg.get("directional_align_overall_min", 0.55))
    switch_band_min = float(cfg.get("switch_band_correct_rate_min", 0.55))
    switch_pre_min = float(cfg.get("switch_pre_correct_rate_min", 0.50))
    switch_post_min = float(cfg.get("switch_post_correct_rate_min", 0.50))
    auc_rel_min = float(cfg.get("auc_switch_rel_min", 0.55))
    directional_align_min_v3 = float(cfg.get("directional_align_overall_min_v3", 0.60))
    switch_band_min_v3 = float(cfg.get("switch_band_correct_rate_min_v3", 0.60))
    switch_margin_min_v3 = float(cfg.get("switch_margin_min_v3", 0.0))
    peak_delay_max = float(cfg.get("peak_delay_max_v3", np.nan))
    if not np.isfinite(peak_delay_max):
        peak_delay_max = float(cfg.get("peak_delay_max_frac_v3", 0.5)) * float(switch_window)
    retained_gap_min_v3 = float(cfg.get("retained_gap_switch_min_v3", 0.10))

    directional_align_pass_v2 = bool(np.isfinite(directional_align_overall) and directional_align_overall >= directional_align_min)
    switch_pre_pass_v2 = bool(np.isfinite(switch_pre_correct_rate) and switch_pre_correct_rate >= switch_pre_min)
    switch_post_pass_v2 = bool(np.isfinite(switch_post_correct_rate) and switch_post_correct_rate >= switch_post_min)
    switch_band_pass_v2 = bool(
        np.isfinite(switch_band_correct_rate)
        and switch_band_correct_rate >= switch_band_min
        and switch_pre_pass_v2
        and switch_post_pass_v2
    )
    switch_auc_pass = bool(np.isfinite(auc_switch_rel) and auc_switch_rel >= auc_rel_min)
    directional_align_pass = bool(
        np.isfinite(directional_align_overall) and directional_align_overall >= directional_align_min_v3
    )
    switch_band_pass = bool(np.isfinite(switch_band_correct_rate) and switch_band_correct_rate >= switch_band_min_v3)
    switch_margin_pass = bool(
        np.isfinite(switch_margin_pre)
        and np.isfinite(switch_margin_post)
        and switch_margin_pre > switch_margin_min_v3
        and switch_margin_post > switch_margin_min_v3
    )
    peak_delay_pass = bool(
        (np.isfinite(peak_delay_lambda) and peak_delay_lambda <= peak_delay_max)
        or (np.isfinite(peak_delay_gate) and peak_delay_gate <= peak_delay_max)
    )
    retained_gap_switch_pass = bool(
        np.isfinite(retained_gap_switch) and retained_gap_switch > retained_gap_min_v3
    )

    out.update(
        {
            "pre_correct_rate": pre_correct_rate,
            "post_correct_rate": post_correct_rate,
            "directional_align_pre": pre_correct_rate,
            "directional_align_post": post_correct_rate,
            "directional_align_overall": directional_align_overall,
            "switch_window": int(switch_window),
            "switch_pre_correct_rate": switch_pre_correct_rate,
            "switch_post_correct_rate": switch_post_correct_rate,
            "switch_band_correct_rate": switch_band_correct_rate,
            "switch_margin_pre": switch_margin_pre,
            "switch_margin_post": switch_margin_post,
            "corr_lambda_regime": corr_lambda_regime,
            "corr_gate_regime": corr_gate_regime,
            "corr_retained_regime": corr_retained_regime,
            "auc_switch_lambda": auc_switch_lambda,
            "auc_switch_gate": auc_switch_gate,
            "auc_switch_rel": auc_switch_rel,
            "retained_gap_switch": retained_gap_switch,
            "peak_delay_lambda": peak_delay_lambda,
            "peak_delay_gate": peak_delay_gate,
            "peak_delay_rel": peak_delay_rel,
            "corr_time_lambda_switch": corr_time_lambda_switch,
            "corr_time_gate_switch": corr_time_gate_switch,
            "corr_time_retained_switch": corr_time_retained_switch,
            "switch_band_pass": switch_band_pass,
            "directional_align_pass": directional_align_pass,
            "switch_margin_pass": switch_margin_pass,
            "peak_delay_pass": peak_delay_pass,
            "retained_gap_switch_pass": retained_gap_switch_pass,
            "directional_align_pass_v2": directional_align_pass_v2,
            "switch_pre_pass_v2": switch_pre_pass_v2,
            "switch_post_pass_v2": switch_post_pass_v2,
            "switch_band_pass_v2": switch_band_pass_v2,
            "switch_auc_pass": switch_auc_pass,
            "switch_nan_reasons": reasons,
        }
    )
    return out


def write_diagnostics_files(out_dir, tag, diagnostics):
    suffix = tag if tag else ""
    base = os.path.join(out_dir, f"step5pp_diagnostics{suffix}")
    json_path = base + ".json"
    csv_path = base + ".csv"
    md_path = base + ".md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    keys = list(diagnostics.keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join([csv_safe(diagnostics.get(k)) for k in keys]) + "\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## Step5++ Diagnostics\n\n")
        f.write("| key | value |\n")
        f.write("| --- | --- |\n")
        for k in keys:
            f.write(f"| {k} | {diagnostics[k]} |\n")


def write_subset_summary_standard(summary_rows, out_dir, tag=""):
    suffix = tag if tag else ""
    keys = [
        "subset",
        "count",
        "mean_lambda",
        "mean_gate_weight",
        "p_active",
        "mean_dist_base",
        "mean_dist_reg0",
        "mean_dist_reg1",
        "mean_retained_ratio",
    ]
    rows = []
    for r in summary_rows:
        rows.append({k: r.get(k) for k in keys})
    csv_path = os.path.join(out_dir, f"subset_summary{suffix}.csv")
    md_path = os.path.join(out_dir, f"subset_summary{suffix}.md")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join([str(r.get(k, "")) for k in keys]) + "\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join([str(r.get(k, "")) for k in keys]) + " |\n")


def write_curve_stats_csv(out_dir, metrics, tag=""):
    suffix = tag if tag else ""
    retained_high = metrics.get("mean_retained_ratio_high")
    retained_low = metrics.get("mean_retained_ratio_low")
    retained_gap = None
    try:
        retained_gap = float(retained_low) - float(retained_high)
    except Exception:
        retained_gap = np.nan
    rows = [
        ("dist_std_base", metrics.get("dist_std_base")),
        ("dist_std_reg0", metrics.get("dist_std_reg0")),
        ("dist_std_reg1", metrics.get("dist_std_reg1")),
        ("rel_pre_mean", metrics.get("rel_pre_mean")),
        ("rel_pre_std", metrics.get("rel_pre_std")),
        ("rel_post_mean", metrics.get("rel_post_mean")),
        ("rel_post_std", metrics.get("rel_post_std")),
        ("retained_high_mean", retained_high),
        ("retained_low_mean", retained_low),
        ("retained_gap", retained_gap),
        ("pre_correct_rate", metrics.get("pre_correct_rate")),
        ("post_correct_rate", metrics.get("post_correct_rate")),
        ("directional_align_overall", metrics.get("directional_align_overall")),
        ("switch_window", metrics.get("switch_window")),
        ("switch_pre_correct_rate", metrics.get("switch_pre_correct_rate")),
        ("switch_post_correct_rate", metrics.get("switch_post_correct_rate")),
        ("switch_band_correct_rate", metrics.get("switch_band_correct_rate")),
        ("switch_margin_pre", metrics.get("switch_margin_pre")),
        ("switch_margin_post", metrics.get("switch_margin_post")),
        ("corr_lambda_regime", metrics.get("corr_lambda_regime")),
        ("corr_gate_regime", metrics.get("corr_gate_regime")),
        ("corr_retained_regime", metrics.get("corr_retained_regime")),
        ("auc_switch_lambda", metrics.get("auc_switch_lambda")),
        ("auc_switch_gate", metrics.get("auc_switch_gate")),
        ("auc_switch_rel", metrics.get("auc_switch_rel")),
        ("retained_gap_switch", metrics.get("retained_gap_switch")),
        ("peak_delay_lambda", metrics.get("peak_delay_lambda")),
        ("peak_delay_gate", metrics.get("peak_delay_gate")),
        ("peak_delay_rel", metrics.get("peak_delay_rel")),
        ("corr_time_lambda_switch", metrics.get("corr_time_lambda_switch")),
        ("corr_time_gate_switch", metrics.get("corr_time_gate_switch")),
        ("corr_time_retained_switch", metrics.get("corr_time_retained_switch")),
        ("switch_margin_pass", metrics.get("switch_margin_pass")),
        ("peak_delay_pass", metrics.get("peak_delay_pass")),
        ("retained_gap_switch_pass", metrics.get("retained_gap_switch_pass")),
        ("switch_band_pass", metrics.get("switch_band_pass")),
        ("directional_align_pass", metrics.get("directional_align_pass")),
        ("pass_core_checks_v2", metrics.get("pass_core_checks_v2")),
        ("pass_core_checks_v3", metrics.get("pass_core_checks_v3")),
    ]
    path = os.path.join(out_dir, f"curve_stats{suffix}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in rows:
            f.write(f"{k},{csv_safe(v)}\n")


def write_checks_json(out_dir, metrics, config_used=None, tag=""):
    suffix = tag if tag else ""
    def b(v):
        return bool(v) if v is not None else False

    data = {
        "gate_direction": b(metrics.get("check_gate_direction")),
        "high_closer_A0": b(metrics.get("check_high_closer_a0")),
        "low_closer_A1": b(metrics.get("check_low_closer_a1")),
        "pre_post_direction": (
            bool(metrics.get("pre_rel_sign_ok")) and bool(metrics.get("post_rel_sign_ok"))
            if (metrics.get("pre_rel_sign_ok") is not None and metrics.get("post_rel_sign_ok") is not None)
            else None
        ),
        "overall_check": b(metrics.get("check_overall_pass")),
        "pre_correct_rate": metrics.get("pre_correct_rate"),
        "post_correct_rate": metrics.get("post_correct_rate"),
        "directional_align_pre": metrics.get("directional_align_pre"),
        "directional_align_post": metrics.get("directional_align_post"),
        "directional_align_overall": metrics.get("directional_align_overall"),
        "switch_window": metrics.get("switch_window"),
        "switch_pre_correct_rate": metrics.get("switch_pre_correct_rate"),
        "switch_post_correct_rate": metrics.get("switch_post_correct_rate"),
        "switch_band_correct_rate": metrics.get("switch_band_correct_rate"),
        "switch_margin_pre": metrics.get("switch_margin_pre"),
        "switch_margin_post": metrics.get("switch_margin_post"),
        "corr_lambda_regime": metrics.get("corr_lambda_regime"),
        "corr_gate_regime": metrics.get("corr_gate_regime"),
        "corr_retained_regime": metrics.get("corr_retained_regime"),
        "auc_switch_lambda": metrics.get("auc_switch_lambda"),
        "auc_switch_gate": metrics.get("auc_switch_gate"),
        "auc_switch_rel": metrics.get("auc_switch_rel"),
        "peak_delay_lambda": metrics.get("peak_delay_lambda"),
        "peak_delay_gate": metrics.get("peak_delay_gate"),
        "peak_delay_rel": metrics.get("peak_delay_rel"),
        "corr_time_lambda_switch": metrics.get("corr_time_lambda_switch"),
        "corr_time_gate_switch": metrics.get("corr_time_gate_switch"),
        "corr_time_retained_switch": metrics.get("corr_time_retained_switch"),
        "retained_gap": metrics.get("retained_gap"),
        "retained_gap_switch": metrics.get("retained_gap_switch"),
        "switch_band_pass": b(metrics.get("switch_band_pass")),
        "directional_align_pass": b(metrics.get("directional_align_pass")),
        "switch_margin_pass": b(metrics.get("switch_margin_pass")),
        "peak_delay_pass": b(metrics.get("peak_delay_pass")),
        "retained_gap_switch_pass": b(metrics.get("retained_gap_switch_pass")),
        "pass_core_checks_v2": b(metrics.get("pass_core_checks_v2")),
        "pass_core_checks_v3": b(metrics.get("pass_core_checks_v3")),
        "switch_nan_reasons": metrics.get("switch_nan_reasons", []),
        "regime_swapped": (config_used or {}).get("regime_swapped"),
        "swap_reason": (config_used or {}).get("swap_reason"),
    }
    with open(os.path.join(out_dir, f"checks{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def write_sanity_metrics_json(out_dir, metrics, summary_rows, config_used=None, tag=""):
    suffix = tag if tag else ""
    def b(v):
        return bool(v) if v is not None else False

    subset_map = {r.get("subset"): r for r in summary_rows}
    high_row = subset_map.get("high_non_sat", subset_map.get("high_sat", {}))
    low_row = subset_map.get("low", {})
    all_row = subset_map.get("all", {})

    data = {
        "config_name": (config_used or {}).get("config_name"),
        "delta_mask_mode": metrics.get("delta_mask_mode"),
        "dist_mask_mode": metrics.get("dist_mask_mode"),
        "delta_mask_nnz": metrics.get("delta_mask_nnz"),
        "dist_mask_nnz": metrics.get("dist_mask_nnz"),
        "A0_eff_nnz": metrics.get("A0_eff_nnz"),
        "A1_eff_nnz": metrics.get("A1_eff_nnz"),
        "subset_strategy": metrics.get("subset_strategy"),
        "high_thr": metrics.get("high_thr"),
        "low_thr": metrics.get("low_thr"),
        "high_non_sat_count": high_row.get("count"),
        "high_non_sat_mean_lambda": high_row.get("mean_lambda"),
        "high_non_sat_mean_gate_weight": high_row.get("mean_gate_weight"),
        "low_count": low_row.get("count"),
        "low_mean_lambda": low_row.get("mean_lambda"),
        "low_mean_gate_weight": low_row.get("mean_gate_weight"),
        "all_count": all_row.get("count"),
        "all_mean_lambda": all_row.get("mean_lambda"),
        "all_mean_gate_weight": all_row.get("mean_gate_weight"),
        "dist_std_base": metrics.get("dist_std_base"),
        "dist_std_reg0": metrics.get("dist_std_reg0"),
        "dist_std_reg1": metrics.get("dist_std_reg1"),
        "align_pre": metrics.get("align_pre"),
        "align_post": metrics.get("align_post"),
        "align_overall": metrics.get("align_overall"),
        "margin_pre": metrics.get("mean_margin_pre"),
        "margin_post": metrics.get("mean_margin_post"),
        "rel_pre_mean": metrics.get("rel_pre_mean"),
        "rel_pre_std": metrics.get("rel_pre_std"),
        "rel_post_mean": metrics.get("rel_post_mean"),
        "rel_post_std": metrics.get("rel_post_std"),
        "mean_dist_reg0_pre": metrics.get("mean_dist_reg0_pre"),
        "mean_dist_reg1_pre": metrics.get("mean_dist_reg1_pre"),
        "mean_dist_reg0_post": metrics.get("mean_dist_reg0_post"),
        "mean_dist_reg1_post": metrics.get("mean_dist_reg1_post"),
        "gate_direction": b(metrics.get("check_gate_direction")),
        "high_closer_A0": b(metrics.get("check_high_closer_a0")),
        "low_closer_A1": b(metrics.get("check_low_closer_a1")),
        "pre_correct_rate": metrics.get("pre_correct_rate"),
        "post_correct_rate": metrics.get("post_correct_rate"),
        "directional_align_pre": metrics.get("directional_align_pre"),
        "directional_align_post": metrics.get("directional_align_post"),
        "directional_align_overall": metrics.get("directional_align_overall"),
        "switch_window": metrics.get("switch_window"),
        "switch_pre_correct_rate": metrics.get("switch_pre_correct_rate"),
        "switch_post_correct_rate": metrics.get("switch_post_correct_rate"),
        "switch_band_correct_rate": metrics.get("switch_band_correct_rate"),
        "switch_margin_pre": metrics.get("switch_margin_pre"),
        "switch_margin_post": metrics.get("switch_margin_post"),
        "corr_lambda_regime": metrics.get("corr_lambda_regime"),
        "corr_gate_regime": metrics.get("corr_gate_regime"),
        "corr_retained_regime": metrics.get("corr_retained_regime"),
        "auc_switch_lambda": metrics.get("auc_switch_lambda"),
        "auc_switch_gate": metrics.get("auc_switch_gate"),
        "auc_switch_rel": metrics.get("auc_switch_rel"),
        "peak_delay_lambda": metrics.get("peak_delay_lambda"),
        "peak_delay_gate": metrics.get("peak_delay_gate"),
        "peak_delay_rel": metrics.get("peak_delay_rel"),
        "corr_time_lambda_switch": metrics.get("corr_time_lambda_switch"),
        "corr_time_gate_switch": metrics.get("corr_time_gate_switch"),
        "corr_time_retained_switch": metrics.get("corr_time_retained_switch"),
        "retained_gap": metrics.get("retained_gap"),
        "retained_gap_switch": metrics.get("retained_gap_switch"),
        "switch_band_pass": b(metrics.get("switch_band_pass")),
        "directional_align_pass": b(metrics.get("directional_align_pass")),
        "switch_margin_pass": b(metrics.get("switch_margin_pass")),
        "peak_delay_pass": b(metrics.get("peak_delay_pass")),
        "retained_gap_switch_pass": b(metrics.get("retained_gap_switch_pass")),
        "pass_core_checks_v2": b(metrics.get("pass_core_checks_v2")),
        "pass_core_checks_v3": b(metrics.get("pass_core_checks_v3")),
        "switch_nan_reasons": metrics.get("switch_nan_reasons", []),
        "pre_post_direction": (
            bool(metrics.get("pre_rel_sign_ok")) and bool(metrics.get("post_rel_sign_ok"))
            if (metrics.get("pre_rel_sign_ok") is not None and metrics.get("post_rel_sign_ok") is not None)
            else None
        ),
        "overall_check": metrics.get("check_overall_pass"),
        "regime_swapped": (config_used or {}).get("regime_swapped"),
        "swap_reason": (config_used or {}).get("swap_reason"),
    }
    with open(os.path.join(out_dir, f"sanity_metrics{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def load_lambda_override(lambda_file, data_dir, logs):
    path = lambda_file
    if not os.path.isabs(path) and not os.path.isfile(path):
        path = os.path.join(data_dir, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"lambda_file not found: {path}")
    if path.lower().endswith(".npz"):
        npz = np.load(path)
        if "lambda_t" in npz:
            lambda_t = np.array(npz["lambda_t"]).reshape(-1)
        else:
            keys = list(npz.keys())
            if not keys:
                raise ValueError(f"lambda npz is empty: {path}")
            lambda_t = np.array(npz[keys[0]]).reshape(-1)
        if "valid_mask" in npz:
            valid_mask = np.array(npz["valid_mask"]).astype(bool).reshape(-1)
        else:
            valid_mask = np.isfinite(lambda_t)
    else:
        lambda_t = np.load(path).reshape(-1)
        valid_mask = np.isfinite(lambda_t)
    lambda_t = np.asarray(lambda_t, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape[0] != lambda_t.shape[0]:
        raise ValueError(f"lambda_file valid_mask length mismatch: {path}")
    lambda_t = np.where(np.isfinite(lambda_t), lambda_t, 0.0)
    logs.append(f"lambda loaded from --lambda_file: {path}")
    return lambda_t, valid_mask, path


def run_step5pp_once(run_once_callable, lambda_t, valid_mask, tag="", write_plots=False, sanity_block=None):
    # Thin wrapper so main/other callers use one entry point and get structured metrics.
    return run_once_callable(
        lambda_t=lambda_t,
        valid_mask=valid_mask,
        tag=tag,
        write_plots=write_plots,
        sanity_block=sanity_block,
    )


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
    parser.add_argument("--lambda_file", type=str, default=None,
                        help="Path to external lambda sequence (.npy or .npz with lambda_t).")
    parser.add_argument("--lambda_tag", type=str, default=None,
                        help="Tag for external lambda source; only used for record.")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    logs = []
    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "exports_step5pp")
    safe_mkdir(out_dir)

    cfg_path = args.config or os.path.join(data_dir, "step5pp_config.json")
    cfg = load_config(cfg_path, data_dir)
    lambda_file_arg = args.lambda_file or cfg.get("lambda_file")
    lambda_tag = args.lambda_tag or cfg.get("lambda_tag")
    validate_cfg(cfg)
    for w in cfg.get("_compat_warnings", []):
        logs.append(f"WARN: {w}")
        print(f"WARN: {w}")
    cfg_summary = (
        f"config: delta_mask_mode={cfg.get('delta_mask_mode')}, "
        f"dist_mask_mode={cfg.get('dist_mask_mode')}, "
        f"auto_swap_regimes={bool(cfg.get('auto_swap_regimes', False))}"
    )
    logs.append(cfg_summary)
    print(cfg_summary)
    meta_path = os.path.join(data_dir, "meta.json")
    t_switch = None
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        t_switch = int(meta.get("t_switch", 0)) if "t_switch" in meta else None

    gate_mode = cfg.get("gate_mode", "soft")
    tau_hard = float(cfg.get("tau_hard", 0.8))
    w_soft = cfg.get("w_soft", None)

    A_base, base_path = load_A_base(data_dir, logs)
    assert_orientation(A_base, "tgt_src")
    regime_ref_source = cfg.get("regime_ref_source", "estimated")
    if regime_ref_source in ("ground_truth", "gt"):
        A0, A1 = load_A0_A1_gt(data_dir, logs)
    else:
        A0, A1 = load_A_regime(data_dir, cfg.get("pred_prefix", "cmiknn"), logs)
    assert_orientation(A0, "tgt_src")
    assert_orientation(A1, "tgt_src")

    X = None
    configs = None
    lambda_source = "unknown"
    lambda_t_single = None
    valid_mask_single = None
    lambda_for_swap = None
    valid_for_swap = None

    if args.score_type:
        x_path = os.path.join(data_dir, "X.npy")
        X = np.load(x_path)
        configs = pick_lambda_configs_from_step4(data_dir, args.score_type, args.top_m)
        lambda_source = f"step4:{args.score_type}:top{args.top_m}"
        if configs:
            lambda_for_swap, valid_for_swap = compute_lambda_kmeans(X, configs[0]["window"], configs[0]["k"])
    else:
        if lambda_file_arg:
            lambda_t_single, valid_mask_single, lambda_source = load_lambda_override(lambda_file_arg, data_dir, logs)
            t_switch_loaded = None
        else:
            lambda_t_single, valid_mask_single, lambda_source, t_switch_loaded = load_lambda_and_mask(data_dir, logs)
        if t_switch_loaded is not None:
            t_switch = int(t_switch_loaded)
        lambda_for_swap = lambda_t_single
        valid_for_swap = valid_mask_single
        if valid_mask_single.sum() > 0:
            v = lambda_t_single[valid_mask_single]
            v = v[np.isfinite(v)]
            if v.size > 0:
                logs.append(f"lambda stats: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

    # true edges from DeltaA
    adj_true, delta_path = find_true_change_from_deltaA(data_dir, logs)
    true_edges = edges_from_adj(adj_true, diag_excluded=True)
    K_true = len(true_edges)
    swap_diag_mask = (adj_true != 0).astype(np.float32)

    A0, A1, regime_swapped, swap_reason = maybe_swap_regimes(
        A0=A0,
        A1=A1,
        t_switch=t_switch,
        lambda_t=lambda_for_swap,
        valid_mask=valid_for_swap,
        cfg=cfg,
        gate_mode=gate_mode,
        tau_hard=tau_hard,
        logs=logs,
        diagnostic_mask=swap_diag_mask,
    )

    logs.append(f"A_base nnz={(np.abs(A_base) > 0).sum()}")
    logs.append(f"A0 stats: min={A0.min():.4f} max={A0.max():.4f} mean={A0.mean():.4f} nnz={(np.abs(A0) > 0).sum()}")
    logs.append(f"A1 stats: min={A1.min():.4f} max={A1.max():.4f} mean={A1.mean():.4f} nnz={(np.abs(A1) > 0).sum()}")
    logs.append(f"regime_swapped={regime_swapped}")
    logs.append(f"swap_reason={swap_reason}")

    # delta proxy
    delta_mode = cfg.get("delta_mode", "A1_minus_A0")
    if delta_mode == "A1_minus_A0":
        delta_proxy = A1 - A0
    else:
        delta_proxy = A1 - A0

    delta_mag = np.abs(delta_proxy)

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

    def run_once(lambda_t, valid_mask, tag, write_plots, sanity_block=None):
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
        delta_mask_mode = cfg.get("delta_mask_mode", "union_base_predchange")
        dist_mask_mode = cfg.get("dist_mask_mode", "true_change_only")
        delta_mask = build_delta_mask(
            mode=delta_mask_mode,
            base_mask=base_mask,
            pred_mask=pred_mask,
            true_mask=true_mask,
            delta_ref=delta_proxy,
            cfg=cfg,
        )

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
        dist_mask = build_dist_mask(
            mode=dist_mask_mode,
            base_mask=base_mask,
            pred_mask=pred_mask,
            true_mask=true_mask,
            delta_ref=(A1_eff - A0_eff),
            cfg=cfg,
            pred_edges=pred_edges,
        )

        nnz_dist = int((dist_mask != 0).sum())
        nnz_delta = int((delta_mask != 0).sum())
        nnz_a0 = int((A0_eff != 0).sum())
        nnz_a1 = int((A1_eff != 0).sum())
        logs.append(f"delta_mask_nnz={nnz_delta} dist_mask_nnz={nnz_dist} tag={tag or 'default'}")
        logs.append(f"A0_eff_nnz={nnz_a0} A1_eff_nnz={nnz_a1} tag={tag or 'default'}")

        # sanity: mask stats
        if args.sanity:
            if sanity_block:
                print(f"=== Sanity Block {sanity_block} ===")
                print(f"config: delta_mask_mode={delta_mask_mode}, dist_mask_mode={dist_mask_mode}")
            diff = (A1_eff - A0_eff) * dist_mask
            diff_abs = np.abs(diff)
            mean_diff = float(diff_abs[dist_mask != 0].mean()) if nnz_dist > 0 else 0.0
            max_diff = float(diff_abs.max()) if nnz_dist > 0 else 0.0
            print(f"mask nnz: delta_mask={nnz_delta} dist_mask={nnz_dist} A0_eff={nnz_a0} A1_eff={nnz_a1}")
            print(f"|A1_eff-A0_eff| on mask: mean={mean_diff:.6f} max={max_diff:.6f}")
            if nnz_dist > 0:
                assert not np.allclose(A0_eff[dist_mask != 0], A1_eff[dist_mask != 0])
            if dist_mask_mode == "true_change_only":
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
        else:
            pre_mask = np.zeros_like(valid_mask, dtype=bool)
            post_mask = np.zeros_like(valid_mask, dtype=bool)
        margin = rel
        check_metrics = evaluate_gate_direction_checks(
            rel=rel,
            pre_mask=pre_mask,
            post_mask=post_mask,
            low_mask=low_mask,
            high_non_sat=high_non_sat,
            gate_weight=gate_weight,
            dist_reg0=dist_reg0,
            dist_reg1=dist_reg1,
        )
        align_all_pre = check_metrics["align_all_pre"]
        align_all_post = check_metrics["align_all_post"]
        overall_align = check_metrics["overall_align"]
        mean_margin_pre = check_metrics["mean_margin_pre"]
        mean_margin_post = check_metrics["mean_margin_post"]
        rel_pre_mean = check_metrics["rel_pre_mean"]
        rel_post_mean = check_metrics["rel_post_mean"]
        rel_pre_std = check_metrics["rel_pre_std"]
        rel_post_std = check_metrics["rel_post_std"]
        mean_dist_reg0_pre = check_metrics["mean_dist_reg0_pre"]
        mean_dist_reg1_pre = check_metrics["mean_dist_reg1_pre"]
        mean_dist_reg0_post = check_metrics["mean_dist_reg0_post"]
        mean_dist_reg1_post = check_metrics["mean_dist_reg1_post"]
        align_low_pre = check_metrics["align_low_pre"]
        align_low_post = check_metrics["align_low_post"]
        align_high_pre = check_metrics["align_high_pre"]
        align_high_post = check_metrics["align_high_post"]

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
        switch_metrics = evaluate_switch_aware_metrics(
            rel=rel,
            lambda_t=lambda_t,
            gate_weight=gate_weight,
            retained_ratio=retained_ratio,
            valid_mask=valid_mask,
            t_switch=t_switch,
            cfg=cfg,
        )
        if switch_metrics.get("switch_nan_reasons"):
            logs.append(
                f"switch-aware nan reasons(tag={tag or 'default'}): "
                + ";".join([str(x) for x in switch_metrics.get("switch_nan_reasons", [])])
            )
        low_post_min = int(cfg.get("low_post_min", 10))
        if n_low_post < low_post_min:
            logs.append("WARN: low_post too small")
            align_low_post = np.nan
            margin_low_post_mean = np.nan
            margin_low_post_std = np.nan
            check_metrics["align_low_post"] = np.nan
            check_metrics["check_low_closer_a1"] = None
            checks = [
                v
                for v in [
                    check_metrics.get("pre_rel_sign_ok"),
                    check_metrics.get("post_rel_sign_ok"),
                ]
                if v is not None
            ]
            check_metrics["check_overall_pass"] = bool(all(checks)) if checks else False

        gate_direction = bool(check_metrics.get("check_gate_direction"))
        high_closer_a0 = bool(check_metrics.get("check_high_closer_a0"))
        low_closer_a1 = bool(check_metrics.get("check_low_closer_a1"))
        base_core_checks = bool(gate_direction and high_closer_a0 and low_closer_a1)
        pass_core_checks_v2 = bool(
            base_core_checks
            and bool(switch_metrics.get("directional_align_pass_v2"))
            and bool(switch_metrics.get("switch_band_pass_v2"))
            and bool(switch_metrics.get("switch_auc_pass"))
        )
        pass_core_checks_v3 = bool(
            base_core_checks
            and bool(switch_metrics.get("directional_align_pass"))
            and bool(switch_metrics.get("switch_band_pass"))
            and bool(switch_metrics.get("switch_margin_pass"))
            and bool(switch_metrics.get("peak_delay_pass"))
            and bool(switch_metrics.get("retained_gap_switch_pass"))
        )

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
        retained_gap = (
            float(mean_retained_low) - float(mean_retained_high)
            if (np.isfinite(mean_retained_low) and np.isfinite(mean_retained_high))
            else np.nan
        )
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
            print(
                "switch-aware: "
                f"pre_correct={switch_metrics.get('pre_correct_rate')} "
                f"post_correct={switch_metrics.get('post_correct_rate')} "
                f"switch_band_correct={switch_metrics.get('switch_band_correct_rate')} "
                f"auc_rel={switch_metrics.get('auc_switch_rel')}"
            )
            print(f"mask: delta_mask_mode={delta_mask_mode} dist_mask_mode={dist_mask_mode}")
            print(f"mask nnz: delta_mask_nnz={nnz_delta} dist_mask_nnz={nnz_dist}")
            print(f"mean(dist_reg0_pre)={mean_dist_reg0_pre:.6f} mean(dist_reg1_pre)={mean_dist_reg1_pre:.6f}")
            print(f"mean(dist_reg0_post)={mean_dist_reg0_post:.6f} mean(dist_reg1_post)={mean_dist_reg1_post:.6f}")
            print(
                "checks: "
                f"gate_direction={check_metrics['check_gate_direction']} "
                f"high_closer_A0={check_metrics['check_high_closer_a0']} "
                f"low_closer_A1={check_metrics['check_low_closer_a1']} "
                f"pre_rel_sign_ok={check_metrics['pre_rel_sign_ok']} "
                f"post_rel_sign_ok={check_metrics['post_rel_sign_ok']} "
                f"overall={check_metrics['check_overall_pass']} "
                f"pass_core_checks_v2={pass_core_checks_v2} "
                f"pass_core_checks_v3={pass_core_checks_v3}"
            )

        if write_plots and not HAS_MPL:
            logs.append("WARN: matplotlib not available; skipping plots.")

        subset_map = {r["subset"]: r for r in summary_rows}
        high_row = subset_map.get("high_non_sat", subset_map.get("high_sat"))
        low_row = subset_map.get("low")
        all_row = subset_map.get("all")
        diagnostics = {
            "subset_high_q": cfg.get("subset_high_q", 0.90),
            "subset_low_q": cfg.get("subset_low_q", 0.50),
            "high_thr": high_thr,
            "low_thr": low_thr,
            "subset_strategy": subset_strategy,
            "count_high": int(high_row["count"]) if high_row else 0,
            "count_low": int(low_row["count"]) if low_row else 0,
            "count_all": int(all_row["count"]) if all_row else int(valid_mask.sum()),
            "mean_lambda_high": float(high_row["mean_lambda"]) if high_row else np.nan,
            "mean_lambda_low": float(low_row["mean_lambda"]) if low_row else np.nan,
            "mean_lambda_all": float(all_row["mean_lambda"]) if all_row else np.nan,
            "mean_gate_high": float(high_row["mean_gate_weight"]) if high_row else np.nan,
            "mean_gate_low": float(low_row["mean_gate_weight"]) if low_row else np.nan,
            "mean_gate_all": float(all_row["mean_gate_weight"]) if all_row else np.nan,
            "dist_std_base": float(dist_base[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "dist_std_reg0": float(dist_reg0[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "dist_std_reg1": float(dist_reg1[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "align_pre": align_all_pre,
            "align_post": align_all_post,
            "align_overall": overall_align,
            "align_all_pre": align_all_pre,
            "align_all_post": align_all_post,
            "align_low_pre": align_low_pre,
            "align_low_post": align_low_post,
            "align_high_pre": align_high_pre,
            "align_high_post": align_high_post,
            "overall_align": overall_align,
            "mean_margin_pre": mean_margin_pre,
            "mean_margin_post": mean_margin_post,
            "rel_pre_mean": rel_pre_mean,
            "rel_pre_std": rel_pre_std,
            "rel_post_mean": rel_post_mean,
            "rel_post_std": rel_post_std,
            "delta_mask_mode": delta_mask_mode,
            "dist_mask_mode": dist_mask_mode,
            "delta_mask_nnz": nnz_delta,
            "dist_mask_nnz": nnz_dist,
            "A0_eff_nnz": nnz_a0,
            "A1_eff_nnz": nnz_a1,
            "n_pre": n_pre,
            "n_post": n_post,
            "n_low": n_low,
            "n_low_pre": n_low_pre,
            "n_low_post": n_low_post,
            "n_high_ns_pre": n_high_ns_pre,
            "n_high_ns_post": n_high_ns_post,
            "mean_dist_reg0_pre": mean_dist_reg0_pre,
            "mean_dist_reg1_pre": mean_dist_reg1_pre,
            "mean_dist_reg0_post": mean_dist_reg0_post,
            "mean_dist_reg1_post": mean_dist_reg1_post,
            "check_gate_direction": check_metrics["check_gate_direction"],
            "check_high_closer_a0": check_metrics["check_high_closer_a0"],
            "check_low_closer_a1": check_metrics["check_low_closer_a1"],
            "pre_rel_sign_ok": check_metrics["pre_rel_sign_ok"],
            "post_rel_sign_ok": check_metrics["post_rel_sign_ok"],
            "pre_post_direction": (
                bool(check_metrics["pre_rel_sign_ok"]) and bool(check_metrics["post_rel_sign_ok"])
                if (check_metrics["pre_rel_sign_ok"] is not None and check_metrics["post_rel_sign_ok"] is not None)
                else None
            ),
            "check_overall_pass": check_metrics["check_overall_pass"],
            "pre_correct_rate": switch_metrics.get("pre_correct_rate"),
            "post_correct_rate": switch_metrics.get("post_correct_rate"),
            "directional_align_pre": switch_metrics.get("directional_align_pre"),
            "directional_align_post": switch_metrics.get("directional_align_post"),
            "directional_align_overall": switch_metrics.get("directional_align_overall"),
            "switch_window": switch_metrics.get("switch_window"),
            "switch_pre_correct_rate": switch_metrics.get("switch_pre_correct_rate"),
            "switch_post_correct_rate": switch_metrics.get("switch_post_correct_rate"),
            "switch_band_correct_rate": switch_metrics.get("switch_band_correct_rate"),
            "switch_margin_pre": switch_metrics.get("switch_margin_pre"),
            "switch_margin_post": switch_metrics.get("switch_margin_post"),
            "corr_lambda_regime": switch_metrics.get("corr_lambda_regime"),
            "corr_gate_regime": switch_metrics.get("corr_gate_regime"),
            "corr_retained_regime": switch_metrics.get("corr_retained_regime"),
            "auc_switch_lambda": switch_metrics.get("auc_switch_lambda"),
            "auc_switch_gate": switch_metrics.get("auc_switch_gate"),
            "auc_switch_rel": switch_metrics.get("auc_switch_rel"),
            "peak_delay_lambda": switch_metrics.get("peak_delay_lambda"),
            "peak_delay_gate": switch_metrics.get("peak_delay_gate"),
            "peak_delay_rel": switch_metrics.get("peak_delay_rel"),
            "corr_time_lambda_switch": switch_metrics.get("corr_time_lambda_switch"),
            "corr_time_gate_switch": switch_metrics.get("corr_time_gate_switch"),
            "corr_time_retained_switch": switch_metrics.get("corr_time_retained_switch"),
            "retained_gap": retained_gap,
            "retained_gap_switch": switch_metrics.get("retained_gap_switch"),
            "switch_band_pass": switch_metrics.get("switch_band_pass"),
            "directional_align_pass": switch_metrics.get("directional_align_pass"),
            "switch_margin_pass": switch_metrics.get("switch_margin_pass"),
            "peak_delay_pass": switch_metrics.get("peak_delay_pass"),
            "retained_gap_switch_pass": switch_metrics.get("retained_gap_switch_pass"),
            "gate_direction": gate_direction,
            "high_closer_A0": high_closer_a0,
            "low_closer_A1": low_closer_a1,
            "pass_core_checks_v2": pass_core_checks_v2,
            "pass_core_checks_v3": pass_core_checks_v3,
            "switch_nan_reasons": switch_metrics.get("switch_nan_reasons", []),
        }
        write_diagnostics_files(out_dir, tag, diagnostics)

        return summary_rows, {
            "subset_strategy": subset_strategy,
            "high_thr": high_thr,
            "low_thr": low_thr,
            "delta_mag_mean": delta_mag_mean,
            "delta_mag_max": delta_mag_max,
            "overall_align": overall_align,
            "mean_retained_ratio_high": mean_retained_high,
            "mean_retained_ratio_low": mean_retained_low,
            "dist_std_base": float(dist_base[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "dist_std_reg0": float(dist_reg0[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "dist_std_reg1": float(dist_reg1[valid_mask].std()) if valid_mask.sum() > 0 else 0.0,
            "align_pre": align_all_pre,
            "align_post": align_all_post,
            "align_overall": overall_align,
            "align_all_pre": align_all_pre,
            "align_all_post": align_all_post,
            "align_low_pre": align_low_pre,
            "align_low_post": align_low_post,
            "align_high_pre": align_high_pre,
            "align_high_post": align_high_post,
            "mean_margin_pre": mean_margin_pre,
            "mean_margin_post": mean_margin_post,
            "rel_pre_mean": rel_pre_mean,
            "rel_pre_std": rel_pre_std,
            "rel_post_mean": rel_post_mean,
            "rel_post_std": rel_post_std,
            "corr_lambda_dist_base": corrcoef_safe(lambda_t, dist_base),
            "corr_gate_dist_base": corrcoef_safe(gate_weight, dist_base),
            "corr_lambda_retained": corrcoef_safe(lambda_t, retained_ratio),
            "pre_correct_rate": switch_metrics.get("pre_correct_rate"),
            "post_correct_rate": switch_metrics.get("post_correct_rate"),
            "directional_align_pre": switch_metrics.get("directional_align_pre"),
            "directional_align_post": switch_metrics.get("directional_align_post"),
            "directional_align_overall": switch_metrics.get("directional_align_overall"),
            "switch_window": switch_metrics.get("switch_window"),
            "switch_pre_correct_rate": switch_metrics.get("switch_pre_correct_rate"),
            "switch_post_correct_rate": switch_metrics.get("switch_post_correct_rate"),
            "switch_band_correct_rate": switch_metrics.get("switch_band_correct_rate"),
            "switch_margin_pre": switch_metrics.get("switch_margin_pre"),
            "switch_margin_post": switch_metrics.get("switch_margin_post"),
            "corr_lambda_regime": switch_metrics.get("corr_lambda_regime"),
            "corr_gate_regime": switch_metrics.get("corr_gate_regime"),
            "corr_retained_regime": switch_metrics.get("corr_retained_regime"),
            "auc_switch_lambda": switch_metrics.get("auc_switch_lambda"),
            "auc_switch_gate": switch_metrics.get("auc_switch_gate"),
            "auc_switch_rel": switch_metrics.get("auc_switch_rel"),
            "peak_delay_lambda": switch_metrics.get("peak_delay_lambda"),
            "peak_delay_gate": switch_metrics.get("peak_delay_gate"),
            "peak_delay_rel": switch_metrics.get("peak_delay_rel"),
            "corr_time_lambda_switch": switch_metrics.get("corr_time_lambda_switch"),
            "corr_time_gate_switch": switch_metrics.get("corr_time_gate_switch"),
            "corr_time_retained_switch": switch_metrics.get("corr_time_retained_switch"),
            "retained_gap": retained_gap,
            "retained_gap_switch": switch_metrics.get("retained_gap_switch"),
            "switch_band_pass": switch_metrics.get("switch_band_pass"),
            "directional_align_pass": switch_metrics.get("directional_align_pass"),
            "switch_margin_pass": switch_metrics.get("switch_margin_pass"),
            "peak_delay_pass": switch_metrics.get("peak_delay_pass"),
            "retained_gap_switch_pass": switch_metrics.get("retained_gap_switch_pass"),
            "gate_direction": gate_direction,
            "high_closer_A0": high_closer_a0,
            "low_closer_A1": low_closer_a1,
            "pass_core_checks_v2": pass_core_checks_v2,
            "pass_core_checks_v3": pass_core_checks_v3,
            "switch_nan_reasons": switch_metrics.get("switch_nan_reasons", []),
            "lambda_stats_pre": lambda_stats_pre,
            "lambda_stats_post": lambda_stats_post,
            "gate_stats_pre": gate_stats_pre,
            "gate_stats_post": gate_stats_post,
            "delta_mask_mode": delta_mask_mode,
            "dist_mask_mode": dist_mask_mode,
            "delta_mask_nnz": nnz_delta,
            "dist_mask_nnz": nnz_dist,
            "A0_eff_nnz": nnz_a0,
            "A1_eff_nnz": nnz_a1,
            "mean_dist_reg0_pre": mean_dist_reg0_pre,
            "mean_dist_reg1_pre": mean_dist_reg1_pre,
            "mean_dist_reg0_post": mean_dist_reg0_post,
            "mean_dist_reg1_post": mean_dist_reg1_post,
            "check_gate_direction": check_metrics["check_gate_direction"],
            "check_high_closer_a0": check_metrics["check_high_closer_a0"],
            "check_low_closer_a1": check_metrics["check_low_closer_a1"],
            "pre_rel_sign_ok": check_metrics["pre_rel_sign_ok"],
            "post_rel_sign_ok": check_metrics["post_rel_sign_ok"],
            "pre_post_direction": (
                bool(check_metrics["pre_rel_sign_ok"]) and bool(check_metrics["post_rel_sign_ok"])
                if (check_metrics["pre_rel_sign_ok"] is not None and check_metrics["post_rel_sign_ok"] is not None)
                else None
            ),
            "check_overall_pass": check_metrics["check_overall_pass"],
            "diagnostics": diagnostics,
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
        if not configs:
            raise ValueError(f"No step4 configs found for score_type={args.score_type}")
        compare_rows = []
        metrics = {}
        for i, c in enumerate(configs, start=1):
            lambda_t, valid_mask = compute_lambda_kmeans(X, c["window"], c["k"])
            tag = f"_{c['window']}_{c['k']}"
            summary_rows, metrics = run_step5pp_once(
                run_once, lambda_t, valid_mask, tag=tag, write_plots=(i == 1)
            )
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
            "config_name": lambda_tag or os.path.splitext(os.path.basename(cfg_path))[0],
            "pred_prefix": cfg.get("pred_prefix", ""),
            "score_type": args.score_type,
            "top_m": args.top_m,
            "run_type": cfg.get("run_type"),
            "control_family": cfg.get("control_family"),
            "lambda_source": lambda_source,
            "lambda_tag": lambda_tag,
            "lambda_file": lambda_file_arg,
            "base_path": base_path,
            "delta_path": delta_path,
            "topk_mode": topk_mode,
            "top_k_source": k_source,
            "delta_mask_mode": metrics.get("delta_mask_mode", cfg.get("delta_mask_mode")),
            "dist_mask_mode": metrics.get("dist_mask_mode", cfg.get("dist_mask_mode")),
            "regime_support_mode": cfg.get("regime_support_mode", "union_base_predchange"),
            "eff_anchor": cfg.get("eff_anchor", cfg.get("eff_base_mode", "A0")),
            "gate_fn": "soft: g=1-lambda" if gate_mode == "soft" else "hard: g=1(lambda<thr)",
            "regime_ref_source": regime_ref_source,
            "auto_swap_regimes": bool(cfg.get("auto_swap_regimes", False)),
            "swap_decision_by": cfg.get("swap_decision_by", "pre_rel_mean"),
            "regime_swapped": regime_swapped,
            "swap_reason": swap_reason,
            "low_post_min": int(cfg.get("low_post_min", 10)),
            "check_overall_pass": metrics.get("check_overall_pass"),
            "pass_core_checks_v2": metrics.get("pass_core_checks_v2"),
            "pass_core_checks_v3": metrics.get("pass_core_checks_v3"),
            "switch_window": metrics.get("switch_window", cfg.get("switch_window")),
        }
        with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(config_used, f, indent=2)
    else:
        lambda_t = lambda_t_single
        valid_mask = valid_mask_single
        summary_rows, metrics = run_step5pp_once(
            run_once, lambda_t, valid_mask, tag="", write_plots=True, sanity_block="A: config dist mask"
        )
        if args.sanity:
            # diagnostic run: focus on true change edges with A0 base
            saved_dist_mask = cfg.get("dist_mask_mode", "true_change_only")
            saved_eff_anchor = cfg.get("eff_anchor", cfg.get("eff_base_mode", "A0"))
            cfg["dist_mask_mode"] = "true_change_only"
            cfg["eff_anchor"] = "A0"
            _, diag = run_step5pp_once(
                run_once, lambda_t, valid_mask, tag="_diag", write_plots=False,
                sanity_block="B: diagnostic true_change_only"
            )
            print(f"diagnostic rel: pre_mean={diag.get('rel_pre_mean')} post_mean={diag.get('rel_post_mean')}")
            cfg["dist_mask_mode"] = saved_dist_mask
            cfg["eff_anchor"] = saved_eff_anchor
        config_used = {
            "data_dir": data_dir,
            "config_path": cfg_path,
            "config_name": lambda_tag or os.path.splitext(os.path.basename(cfg_path))[0],
            "lambda_source": lambda_source,
            "lambda_tag": lambda_tag,
            "lambda_file": lambda_file_arg,
            "base_path": base_path,
            "delta_path": delta_path,
            "pred_prefix": cfg.get("pred_prefix", ""),
            "score_type": cfg.get("score_type", ""),
            "run_type": cfg.get("run_type"),
            "control_family": cfg.get("control_family"),
            "delta_mode": delta_mode,
            "gate_mode": gate_mode,
            "gate_fn": "soft: g=1-lambda" if gate_mode == "soft" else "hard: g=1(lambda<thr)",
            "tau_hard": tau_hard,
            "w_soft": w_soft,
            "subset_high_q": cfg.get("subset_high_q", 0.90),
            "subset_low_q": cfg.get("subset_low_q", 0.50),
            "delta_mask_mode": metrics.get("delta_mask_mode"),
            "dist_mask_mode": metrics.get("dist_mask_mode"),
            "regime_support_mode": cfg.get("regime_support_mode", "union_base_predchange"),
            "eff_anchor": cfg.get("eff_anchor", cfg.get("eff_base_mode", "A0")),
            "regime_ref_source": regime_ref_source,
            "auto_swap_regimes": bool(cfg.get("auto_swap_regimes", False)),
            "swap_decision_by": cfg.get("swap_decision_by", "pre_rel_mean"),
            "regime_swapped": regime_swapped,
            "swap_reason": swap_reason,
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
            "delta_mask_nnz": metrics.get("delta_mask_nnz"),
            "dist_mask_nnz": metrics.get("dist_mask_nnz"),
            "check_gate_direction": metrics.get("check_gate_direction"),
            "check_high_closer_a0": metrics.get("check_high_closer_a0"),
            "check_low_closer_a1": metrics.get("check_low_closer_a1"),
            "pre_rel_sign_ok": metrics.get("pre_rel_sign_ok"),
            "post_rel_sign_ok": metrics.get("post_rel_sign_ok"),
            "pre_post_direction": (
                bool(metrics.get("pre_rel_sign_ok")) and bool(metrics.get("post_rel_sign_ok"))
                if (metrics.get("pre_rel_sign_ok") is not None and metrics.get("post_rel_sign_ok") is not None)
                else None
            ),
            "check_overall_pass": metrics.get("check_overall_pass"),
            "pass_core_checks_v2": metrics.get("pass_core_checks_v2"),
            "pass_core_checks_v3": metrics.get("pass_core_checks_v3"),
            "pre_correct_rate": metrics.get("pre_correct_rate"),
            "post_correct_rate": metrics.get("post_correct_rate"),
            "directional_align_overall": metrics.get("directional_align_overall"),
            "switch_band_correct_rate": metrics.get("switch_band_correct_rate"),
            "auc_switch_rel": metrics.get("auc_switch_rel"),
            "switch_window": metrics.get("switch_window"),
            "peak_delay_lambda": metrics.get("peak_delay_lambda"),
            "peak_delay_gate": metrics.get("peak_delay_gate"),
            "peak_delay_rel": metrics.get("peak_delay_rel"),
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
        write_subset_summary_standard(summary_rows, out_dir, tag="")
        write_curve_stats_csv(out_dir, metrics, tag="")
        write_checks_json(out_dir, metrics, config_used=config_used, tag="")
        write_sanity_metrics_json(out_dir, metrics, summary_rows, config_used=config_used, tag="")
        sweep_raw = cfg.get("switch_window_sweep", [])
        sweep_vals = []
        if isinstance(sweep_raw, (list, tuple)):
            for x in sweep_raw:
                try:
                    sweep_vals.append(max(1, int(x)))
                except Exception:
                    logs.append(f"WARN: invalid switch_window_sweep value ignored: {x}")
        sweep_vals = sorted(set(sweep_vals))
        if sweep_vals:
            orig_switch_window = int(cfg.get("switch_window", 200))
            for sw in sweep_vals:
                if sw == orig_switch_window:
                    continue
                cfg["switch_window"] = int(sw)
                tag = f"_sw{int(sw)}"
                summary_rows_sw, metrics_sw = run_step5pp_once(
                    run_once, lambda_t, valid_mask, tag=tag, write_plots=False
                )
                config_used_sw = dict(config_used)
                config_used_sw["switch_window"] = int(sw)
                config_used_sw["pass_core_checks_v2"] = metrics_sw.get("pass_core_checks_v2")
                config_used_sw["pass_core_checks_v3"] = metrics_sw.get("pass_core_checks_v3")
                write_subset_summary_standard(summary_rows_sw, out_dir, tag=tag)
                write_curve_stats_csv(out_dir, metrics_sw, tag=tag)
                write_checks_json(out_dir, metrics_sw, config_used=config_used_sw, tag=tag)
                write_sanity_metrics_json(out_dir, metrics_sw, summary_rows_sw, config_used=config_used_sw, tag=tag)
            cfg["switch_window"] = orig_switch_window

    # Sanity header in logs
    header = [
        "=== Sanity Header ===",
        f"data_dir={data_dir}",
        f"config_path={cfg_path}",
        f"gate_fn={'soft: g=1-lambda' if gate_mode == 'soft' else 'hard: g=1(lambda<thr)'}",
    ]
    if not args.score_type:
        header.extend([
            "rel = dist_reg0 - dist_reg1; rel>0 => closer to A1",
            f"delta_mask_mode={config_used.get('delta_mask_mode')}",
            f"dist_mask_mode={config_used.get('dist_mask_mode')}",
            f"regime_support_mode={config_used.get('regime_support_mode')}",
            f"eff_anchor={config_used.get('eff_anchor')}",
            f"regime_ref_source={config_used.get('regime_ref_source')}",
            f"regime_swapped={config_used.get('regime_swapped')}",
            f"swap_reason={config_used.get('swap_reason')}",
            f"topk_mode={config_used.get('topk_mode')}",
            f"k_source={config_used.get('top_k_source')}",
            f"lambda_stats_pre={config_used.get('lambda_stats_pre')}",
            f"lambda_stats_post={config_used.get('lambda_stats_post')}",
            f"gate_stats_pre={config_used.get('gate_stats_pre')}",
            f"gate_stats_post={config_used.get('gate_stats_post')}",
            f"check_overall_pass={config_used.get('check_overall_pass')}",
            f"pass_core_checks_v2={config_used.get('pass_core_checks_v2')}",
            f"pass_core_checks_v3={config_used.get('pass_core_checks_v3')}",
            f"switch_window={config_used.get('switch_window')}",
        ])
    logs = header + logs
    write_logs(logs, os.path.join(out_dir, "logs.txt"))

    if args.sanity:
        print("=== Step5++ sanity ===")
        print(f"A_base nnz={int((np.abs(A_base) > 0).sum())}")
        print(f"A0 min/max/mean: {A0.min():.4f}/{A0.max():.4f}/{A0.mean():.4f}")
        print(f"A1 min/max/mean: {A1.min():.4f}/{A1.max():.4f}/{A1.mean():.4f}")
        print(f"regime_swapped={regime_swapped} reason={swap_reason}")
        print(f"K_true={K_true}, K_pred={len(pred_edges)} (source={k_source})")


if __name__ == "__main__":
    main()
