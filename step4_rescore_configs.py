import os
import csv
import json
import argparse
from datetime import datetime
import hashlib

import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from step5pp_utils import compute_lambda_kmeans


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


def find_data_dir_from_inputs(in_csvs, explicit_data_dir):
    if explicit_data_dir:
        return explicit_data_dir
    for p in in_csvs:
        ap = os.path.abspath(p)
        parts = ap.split(os.sep)
        if "exports_step4" in parts:
            i = parts.index("exports_step4")
            if i > 0:
                return os.sep.join(parts[:i])
    return None


def read_t_switch(data_dir, t_switch_arg):
    if t_switch_arg is not None:
        return int(t_switch_arg)
    if not data_dir:
        return None
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "t_switch" in meta:
            return int(meta["t_switch"])
    return None


def orient_signal_post_high(signal, valid_mask, t_switch):
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    v = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(s)
    t = np.arange(s.size, dtype=int)
    pre = v & (t < int(t_switch))
    post = v & (t >= int(t_switch))
    pre_mean = float(np.mean(s[pre])) if int(pre.sum()) > 0 else np.nan
    post_mean = float(np.mean(s[post])) if int(post.sum()) > 0 else np.nan
    swapped = bool(np.isfinite(pre_mean) and np.isfinite(post_mean) and (post_mean < pre_mean))
    s_used = -s if swapped else s
    return s_used, swapped, pre_mean, post_mean


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


def lambda_hash_round6(lambda_t, valid_mask):
    arr = np.asarray(lambda_t, dtype=np.float64).reshape(-1)
    m = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(arr)
    if int(m.sum()) == 0:
        return ""
    payload = np.round(arr[m], 6).astype(np.float64).tobytes()
    return hashlib.sha1(payload).hexdigest()


def compute_switch_metrics(lambda_t, valid_mask, t_switch, switch_window):
    t = np.arange(len(lambda_t), dtype=int)
    sig, swapped, pre_mean, post_mean = orient_signal_post_high(lambda_t, valid_mask, t_switch)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(sig)
    pre = valid & (t < int(t_switch))
    post = valid & (t >= int(t_switch))
    pre_local = valid & (t >= int(t_switch) - int(switch_window)) & (t < int(t_switch))
    post_local = valid & (t >= int(t_switch)) & (t < int(t_switch) + int(switch_window))
    band = pre_local | post_local

    if int(pre_local.sum()) > 0 and int(post_local.sum()) > 0:
        mean_pre = float(np.mean(sig[pre_local]))
        mean_post = float(np.mean(sig[post_local]))
        thr = 0.5 * (mean_pre + mean_post)
    elif int(band.sum()) > 0:
        mean_pre = np.nan
        mean_post = np.nan
        thr = float(np.median(sig[band]))
    else:
        mean_pre = np.nan
        mean_post = np.nan
        thr = np.nan

    pre_correct = float(np.mean(sig[pre_local] < thr)) if int(pre_local.sum()) > 0 and np.isfinite(thr) else np.nan
    post_correct = float(np.mean(sig[post_local] >= thr)) if int(post_local.sum()) > 0 and np.isfinite(thr) else np.nan
    if int(pre_local.sum()) + int(post_local.sum()) > 0 and np.isfinite(thr):
        n = float(int(pre_local.sum()) + int(post_local.sum()))
        band_correct = float((np.sum(sig[pre_local] < thr) + np.sum(sig[post_local] >= thr)) / n)
    else:
        band_correct = np.nan

    margin_pre = float(thr - mean_pre) if np.isfinite(thr) and np.isfinite(mean_pre) else np.nan
    margin_post = float(mean_post - thr) if np.isfinite(thr) and np.isfinite(mean_post) else np.nan
    margin_gap = float(min(margin_pre, margin_post)) if np.isfinite(margin_pre) and np.isfinite(margin_post) else np.nan

    idx = np.arange(len(sig), dtype=int)
    far_step = (
        valid[1:] & valid[:-1] &
        (np.abs(idx[1:] - int(t_switch)) > int(switch_window)) &
        (np.abs(idx[:-1] - int(t_switch)) > int(switch_window))
    )
    d = sig[1:] - sig[:-1]
    smooth_mask_count = int(far_step.sum())
    if smooth_mask_count > 0:
        smooth_mean_abs_far = float(np.abs(d[far_step]).mean())
        smooth_std_far = float(np.std(d[far_step]))
    else:
        smooth_mean_abs_far = np.nan
        smooth_std_far = np.nan

    return {
        "lambda_oriented_swapped": swapped,
        "lambda_mean_pre": pre_mean,
        "lambda_mean_post": post_mean,
        "switch_pre_correct_rate": pre_correct,
        "switch_post_correct_rate": post_correct,
        "switch_band_correct_rate": band_correct,
        "switch_margin_pre_signed": margin_pre,
        "switch_margin_post_signed": margin_post,
        "switch_margin_gap_signed": margin_gap,
        "smooth_mean_abs_diff_far": smooth_mean_abs_far,
        "smooth_std_diff_far": smooth_std_far,
        "smooth_mask_count": smooth_mask_count,
        "switch_band_count": int(band.sum()),
        "oriented_signal": sig,
        "valid_signal_mask": valid,
    }


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
        "switch_band_correct_rate", "switch_margin_gap_signed", "peak_delay_lambda",
        "smooth_mean_abs_diff_used", "smooth_std_diff_used"
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
            f"{float(r.get('switch_band_correct_rate', np.nan)):.6f}",
            f"{float(r.get('switch_margin_gap_signed', np.nan)):.6f}",
            f"{float(r.get('peak_delay_lambda', np.nan)):.6f}",
            f"{float(r.get('smooth_mean_abs_diff_used', np.nan)):.6f}",
            f"{float(r.get('smooth_std_diff_used', np.nan)):.6f}",
        ]) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def topk_table(rows, score_key, k=5):
    ordered = sorted(rows, key=lambda r: (r.get(score_key) if np.isfinite(r.get(score_key, np.nan)) else -1e9), reverse=True)
    return ordered[:k]


def plot_component_contrib(row, score_name, weights, out_path):
    if plt is None:
        return
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
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--t_switch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--switch_window", type=int, default=200)
    parser.add_argument("--peak_delay_base_window", type=int, default=200)
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

    # switch-aware diagnostics (optional but enabled by default when data/t_switch are available).
    data_dir = find_data_dir_from_inputs(in_csvs, args.data_dir)
    x_path = os.path.join(data_dir, "X.npy") if data_dir else None
    t_switch = read_t_switch(data_dir, args.t_switch)
    switch_window = int(args.switch_window)
    peak_delay_base_window = int(args.peak_delay_base_window)
    x_ok = bool(x_path and os.path.isfile(x_path))
    switch_ready = bool(x_ok and (t_switch is not None))
    lambda_cache = {}
    if switch_ready:
        X = np.load(x_path)
        print(f"[INFO] switch-aware enabled: data_dir={data_dir}, t_switch={int(t_switch)}, switch_window={switch_window}")
        print(f"[INFO] peak_delay uses base_window={peak_delay_base_window}")
    else:
        X = None
        print("[WARN] switch-aware fields will be NaN (missing data_dir/X.npy or t_switch).")

    def get_lambda(window, k):
        key = (int(window), int(k))
        if key in lambda_cache:
            return lambda_cache[key]
        lam, vm = compute_lambda_kmeans(X, window=int(window), k=int(k), seed=int(args.seed))
        lambda_cache[key] = (lam, vm)
        return lam, vm

    for r in rows:
        r["switch_pre_correct_rate"] = np.nan
        r["switch_post_correct_rate"] = np.nan
        r["switch_band_correct_rate"] = np.nan
        r["switch_margin_pre_signed"] = np.nan
        r["switch_margin_post_signed"] = np.nan
        r["switch_margin_gap_signed"] = np.nan
        r["peak_delay_lambda"] = np.nan
        r["lambda_hash_round6"] = ""
        r["smooth_mean_abs_diff_far"] = np.nan
        r["smooth_std_diff_far"] = np.nan
        r["smooth_mask_count"] = 0
        r["lambda_oriented_swapped"] = False

        if not switch_ready:
            continue
        w = r.get("window")
        k = r.get("k")
        if w is None or k is None:
            continue
        lam, vm = get_lambda(w, k)
        r["lambda_hash_round6"] = lambda_hash_round6(lam, vm)
        sw = compute_switch_metrics(lam, vm, int(t_switch), int(switch_window))
        r["switch_pre_correct_rate"] = sw["switch_pre_correct_rate"]
        r["switch_post_correct_rate"] = sw["switch_post_correct_rate"]
        r["switch_band_correct_rate"] = sw["switch_band_correct_rate"]
        r["switch_margin_pre_signed"] = sw["switch_margin_pre_signed"]
        r["switch_margin_post_signed"] = sw["switch_margin_post_signed"]
        r["switch_margin_gap_signed"] = sw["switch_margin_gap_signed"]
        r["smooth_mean_abs_diff_far"] = sw["smooth_mean_abs_diff_far"]
        r["smooth_std_diff_far"] = sw["smooth_std_diff_far"]
        r["smooth_mask_count"] = sw["smooth_mask_count"]
        r["lambda_oriented_swapped"] = bool(sw["lambda_oriented_swapped"])

        base_lam, base_vm = get_lambda(peak_delay_base_window, k)
        base_oriented, _, _, _ = orient_signal_post_high(base_lam, base_vm, int(t_switch))
        r["peak_delay_lambda"] = peak_delay_switch(base_oriented, base_vm, int(t_switch), int(switch_window))
        r["peak_delay_base_window"] = int(peak_delay_base_window)

    smooth_counts = np.array([to_float(r.get("smooth_mask_count")) for r in rows], dtype=float)
    if np.isfinite(smooth_counts).any():
        valid_counts = smooth_counts[np.isfinite(smooth_counts)]
        print(
            "[INFO] smooth_mask_count stats: "
            f"min={int(valid_counts.min())}, med={int(np.median(valid_counts))}, max={int(valid_counts.max())}"
        )

    for r in rows:
        sm_far = to_float(r.get("smooth_mean_abs_diff_far"))
        ss_far = to_float(r.get("smooth_std_diff_far"))
        r["smooth_mean_abs_diff_used"] = float(sm_far) if np.isfinite(sm_far) else float(r.get("smooth_mean_abs_diff"))
        r["smooth_std_diff_used"] = float(ss_far) if np.isfinite(ss_far) else float(r.get("smooth_std_diff"))

    # normalization
    norm_fn = robust_minmax if args.norm_mode == "robust_minmax" else robust_z
    n_auc = norm_fn([r["auc_regime"] for r in rows], args.norm_lo, args.norm_hi)
    n_corr = norm_fn([r["corr_mse"] for r in rows], args.norm_lo, args.norm_hi)
    n_sep = norm_fn([r["sep_mean"] for r in rows], args.norm_lo, args.norm_hi)
    n_top10 = norm_fn([r["top10_in_reg1"] for r in rows], args.norm_lo, args.norm_hi)
    n_smooth_mean = norm_fn([r["smooth_mean_abs_diff_used"] for r in rows], args.norm_lo, args.norm_hi)
    n_smooth_std = norm_fn([r["smooth_std_diff_used"] for r in rows], args.norm_lo, args.norm_hi)
    n_switch_band = norm_fn([r["switch_band_correct_rate"] for r in rows], args.norm_lo, args.norm_hi)
    n_switch_margin = norm_fn([r["switch_margin_gap_signed"] for r in rows], args.norm_lo, args.norm_hi)
    n_peak_delay_raw = norm_fn([r["peak_delay_lambda"] for r in rows], args.norm_lo, args.norm_hi)
    n_peak_delay = 1.0 - np.asarray(n_peak_delay_raw, dtype=float)

    for i, r in enumerate(rows):
        r["n_auc"] = float(n_auc[i])
        r["n_corr"] = float(n_corr[i])
        r["n_sep"] = float(n_sep[i])
        r["n_top10"] = float(n_top10[i])
        r["n_smooth_mean"] = float(n_smooth_mean[i])
        r["n_smooth_std"] = float(n_smooth_std[i])
        r["n_switch_band"] = float(n_switch_band[i])
        r["n_switch_margin_gap"] = float(n_switch_margin[i])
        r["n_peak_delay"] = float(n_peak_delay[i])

    # filters
    filtered_out = []
    kept = []
    if args.no_filters:
        for r in rows:
            r["passed_filters"] = True
            kept.append(r)
    else:
        smooth_mean_vals = np.array([r["smooth_mean_abs_diff_used"] for r in rows], dtype=float)
        smooth_std_vals = np.array([r["smooth_std_diff_used"] for r in rows], dtype=float)
        if args.smooth_mean_max is not None:
            smooth_mean_thr = float(args.smooth_mean_max)
        else:
            smooth_mean_thr = float(np.quantile(smooth_mean_vals[np.isfinite(smooth_mean_vals)], args.smooth_mean_q))
        if args.smooth_std_max is not None:
            smooth_std_thr = float(args.smooth_std_max)
        else:
            smooth_std_thr = float(np.quantile(smooth_std_vals[np.isfinite(smooth_std_vals)], args.smooth_std_q))

        corr_exception_min = -0.01
        switch_band_exception_min = 0.80
        switch_margin_exception_min = 0.20
        regime_corr_exception_min = -0.06
        regime_switch_band_exception_min = 0.85
        regime_switch_margin_exception_min = 0.20
        regime_peak_delay_exception_max = 171.0

        for r in rows:
            reasons = []
            sm = r["smooth_mean_abs_diff_used"]
            ss = r["smooth_std_diff_used"]
            cm = r["corr_mse"]
            vr = r.get("valid_ratio", np.nan)
            sb = r.get("switch_band_correct_rate", np.nan)
            smg = r.get("switch_margin_gap_signed", np.nan)
            pd = r.get("peak_delay_lambda", np.nan)

            corr_soft_exception_base = bool(
                np.isfinite(cm) and
                np.isfinite(sb) and
                np.isfinite(smg) and
                np.isfinite(sm) and
                np.isfinite(ss) and
                (cm >= corr_exception_min) and
                (sb >= switch_band_exception_min) and
                (smg >= switch_margin_exception_min) and
                (sm <= smooth_mean_thr) and
                (ss <= smooth_std_thr)
            )
            corr_soft_exception_regime = bool(
                np.isfinite(cm) and
                np.isfinite(sb) and
                np.isfinite(smg) and
                np.isfinite(pd) and
                np.isfinite(sm) and
                np.isfinite(ss) and
                (cm >= regime_corr_exception_min) and
                (sb >= regime_switch_band_exception_min) and
                (smg >= regime_switch_margin_exception_min) and
                (pd <= regime_peak_delay_exception_max) and
                (sm <= smooth_mean_thr) and
                (ss <= smooth_std_thr)
            )
            corr_soft_exception = bool(corr_soft_exception_base or corr_soft_exception_regime)
            exception_type = ""
            if corr_soft_exception_base:
                exception_type = "base"
            elif corr_soft_exception_regime:
                exception_type = "regime_v2"
            r["corr_mse_soft_exception"] = corr_soft_exception
            r["corr_mse_soft_exception_type"] = exception_type

            if not np.isfinite(sm) or sm > smooth_mean_thr:
                reasons.append("smooth_mean_abs_diff")
            if not np.isfinite(ss) or ss > smooth_std_thr:
                reasons.append("smooth_std_diff")
            if (not np.isfinite(cm) or cm < 0) and (not corr_soft_exception):
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
    # Phase B / iteration 1:
    # Keep the candidate space fixed and only bias regime scoring toward
    # switch-local alignment. The goal is not to maximize global correlation,
    # but to prefer regime lambdas that are sharper around t_switch while
    # still keeping a weak far-region smoothness regularizer.
    regime_score_weights = {
        "auc": 0.22,
        "corr": 0.10,
        "sep": 0.08,
        "top10": 0.05,
        "smooth_mean": 0.04,
        "smooth_std": 0.03,
        "switch_band": 0.27,
        "switch_margin": 0.17,
        "peak_delay": 0.04,
    }
    for r in rows:
        n_auc_v = r["n_auc"]
        n_corr_v = r["n_corr"]
        n_sep_v = r["n_sep"]
        n_top10_v = r["n_top10"]
        n_sm = r["n_smooth_mean"]
        n_ss = r["n_smooth_std"]
        n_switch_band_v = r["n_switch_band"]
        n_switch_margin_v = r["n_switch_margin_gap"]
        n_peak_delay_v = r["n_peak_delay"]
        r["score_equal"] = float(
            n_auc_v + (1 - n_sm) + n_top10_v + n_corr_v + n_sep_v + (1 - n_ss) +
            n_switch_band_v + n_switch_margin_v + n_peak_delay_v
        )
        r["score_gating"] = float(
            0.40 * n_corr_v +
            0.12 * n_auc_v +
            0.08 * n_top10_v +
            0.08 * n_sep_v +
            0.05 * (1 - n_sm) +
            0.05 * (1 - n_ss) +
            0.10 * n_switch_band_v +
            0.07 * n_switch_margin_v +
            0.05 * n_peak_delay_v
        )
        r["score_regime"] = float(
            regime_score_weights["auc"] * n_auc_v +
            regime_score_weights["sep"] * n_sep_v +
            regime_score_weights["corr"] * n_corr_v +
            regime_score_weights["top10"] * n_top10_v +
            regime_score_weights["smooth_mean"] * (1 - n_sm) +
            regime_score_weights["smooth_std"] * (1 - n_ss) +
            regime_score_weights["switch_band"] * n_switch_band_v +
            regime_score_weights["switch_margin"] * n_switch_margin_v +
            regime_score_weights["peak_delay"] * n_peak_delay_v
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rescored_csv = os.path.join(out_dir, f"rescored_results_{ts}.csv")
    rescored_md = os.path.join(out_dir, f"rescored_results_{ts}.md")
    rescore_csv = os.path.join(out_dir, f"rescore_results_{ts}.csv")
    rescore_md = os.path.join(out_dir, f"rescore_results_{ts}.md")
    write_csv(kept, rescored_csv)
    write_md(kept, rescored_md, title="Rescored Results")
    write_csv(kept, rescore_csv)
    write_md(kept, rescore_md, title="Rescore Results")

    if len(in_csvs) > 1:
        merged_path = os.path.join(out_dir, f"merged_rescored_results_{ts}.csv")
        merged_path_alias = os.path.join(out_dir, f"merged_rescore_results_{ts}.csv")
        write_csv(kept, merged_path)
        write_csv(kept, merged_path_alias)
    else:
        merged_path = None
        merged_path_alias = None

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
    if plt is not None:
        smooth = np.array([r.get("smooth_mean_abs_diff_used") for r in kept], dtype=float)
        aucs = np.array([r.get("auc_regime") for r in kept], dtype=float)
        corrs = np.array([r.get("corr_mse") for r in kept], dtype=float)
        scores = np.array([r.get("score_gating") for r in kept], dtype=float)

        pareto1 = os.path.join(out_dir, "pareto_auc_vs_smooth.png")
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        sc = ax.scatter(smooth, aucs, c=scores, cmap="viridis", s=35, edgecolors="none")
        ax.set_xlabel("smooth_mean_abs_diff_used (lower is better)")
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
        ax.set_xlabel("smooth_mean_abs_diff_used (lower is better)")
        ax.set_ylabel("corr_mse (higher is better)")
        ax.set_title("Pareto: Corr vs Smooth")
        fig.colorbar(sc, ax=ax, shrink=0.85, label="score_gating")
        fig.tight_layout()
        fig.savefig(pareto2, dpi=200)
        plt.close(fig)
    else:
        print("[WARN] matplotlib is unavailable, skip contribution/pareto plotting.")

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
    print(f"[OK] Saved: {rescore_csv}")
    print(f"[OK] Saved: {rescore_md}")
    if merged_path:
        print(f"[OK] Saved: {merged_path}")
    if merged_path_alias:
        print(f"[OK] Saved: {merged_path_alias}")
    if filtered_out:
        print(f"[OK] Saved: {filtered_path}")
    print(f"[OK] Saved: {top_txt}")
    print(f"[OK] Saved: {report_path}")
    print("Top-5 consistency:", sorted(list(common_top5)))


if __name__ == "__main__":
    main()
