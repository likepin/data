import os
import csv
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def find_x_npy(data_dir):
    direct = os.path.join(data_dir, "X.npy")
    if os.path.isfile(direct):
        return direct
    for root, _, files in os.walk(data_dir):
        if "X.npy" in files:
            return os.path.join(root, "X.npy")
    raise FileNotFoundError(f"X.npy not found under: {data_dir}")


def read_t_switch(data_dir, t_switch_arg):
    if t_switch_arg is not None:
        return int(t_switch_arg)
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "t_switch" in meta:
            return int(meta["t_switch"])
    return None


def parse_int_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def skewness(x, eps=1e-8):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    centered = x - mean
    m3 = (centered ** 3).mean(axis=0)
    return m3 / (std ** 3 + eps)


def build_window_features(X, window):
    T, N = X.shape
    valid_mask = np.zeros(T, dtype=bool)
    feats = []
    indices = []
    for t in range(T):
        if t < window - 1:
            continue
        w = X[t - window + 1:t + 1]
        mean = w.mean(axis=0)
        std = w.std(axis=0)
        skw = skewness(w)
        rng = w.max(axis=0) - w.min(axis=0)
        feat = np.concatenate([mean, std, skw, rng], axis=0)
        feats.append(feat)
        indices.append(t)
        valid_mask[t] = True
    if len(feats) == 0:
        return np.zeros((0, 4 * X.shape[1]), dtype=np.float64), np.array([], dtype=np.int64), valid_mask
    return np.vstack(feats).astype(np.float64), np.array(indices, dtype=np.int64), valid_mask


def standardize_features(F, eps=1e-8):
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (F - mu) / sd


def kmeans_sklearn(F, k, seed):
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return None
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(F)
    centers = model.cluster_centers_
    return labels, centers


def kmeans_simple(F, k, seed, max_iter=100, tol=1e-4):
    rng = np.random.RandomState(seed)
    n, d = F.shape
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, d), dtype=np.float64)
    if k > n:
        k = n
    init_idx = rng.choice(n, size=k, replace=False if n >= k else True)
    centers = F[init_idx].copy()
    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(max_iter):
        d2 = ((F[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = new_labels == j
            if mask.any():
                new_centers[j] = F[mask].mean(axis=0)
            else:
                new_centers[j] = F[rng.randint(0, n)]
        shift = np.sqrt(((centers - new_centers) ** 2).sum(axis=1)).mean()
        centers = new_centers
        labels = new_labels
        if shift < tol:
            break
    return labels, centers


def nearest_center_distance(F, centers):
    if F.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    d2 = ((F[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def quantile_normalize_p10_p90(values, eps=1e-12):
    if values.size == 0:
        return values.copy()
    q10 = np.quantile(values, 0.10)
    q90 = np.quantile(values, 0.90)
    if not np.isfinite(q10) or not np.isfinite(q90) or q90 <= q10 + eps:
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin + eps:
            return np.zeros_like(values)
        out = (values - vmin) / (vmax - vmin)
        return np.clip(out, 0.0, 1.0)
    out = (values - q10) / (q90 - q10)
    return np.clip(out, 0.0, 1.0)


def auc_roc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    mask = np.isfinite(scores)
    y_true = y_true[mask]
    scores = scores[mask]
    n = scores.size
    if n == 0:
        return np.nan
    n_pos = int((y_true == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def corrcoef_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def ridge_fit(X, Y, alpha=1.0):
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    X_aug = np.concatenate([X, ones], axis=1)
    d = X_aug.shape[1]
    XtX = X_aug.T @ X_aug
    reg = alpha * np.eye(d, dtype=X_aug.dtype)
    XtY = X_aug.T @ Y
    W = np.linalg.solve(XtX + reg, XtY)
    return W


def ridge_predict(X, W):
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    X_aug = np.concatenate([X, ones], axis=1)
    return X_aug @ W


def make_bucket_edges(values, n_buckets):
    qs = np.linspace(0.0, 1.0, n_buckets + 1)
    return np.quantile(values, qs)


def bucket_assign(values, edges):
    b = len(edges) - 1
    out = np.full(values.shape, -1, dtype=np.int64)
    for i in range(b):
        lo = edges[i]
        hi = edges[i + 1]
        if i == b - 1:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        out[mask] = i
    return out


def compute_bucket_stats(lam, mse, mae, n_buckets=5):
    edges = make_bucket_edges(lam, n_buckets)
    bidx = bucket_assign(lam, edges)
    rows = []
    for b in range(n_buckets):
        sel = bidx == b
        count = int(sel.sum())
        if count == 0:
            rows.append({
                "bucket": b,
                "count": 0,
                "lambda_min": float(edges[b]),
                "lambda_max": float(edges[b + 1]),
                "mse_mean": np.nan,
                "mse_std": np.nan,
                "mae_mean": np.nan,
                "mae_std": np.nan,
            })
            continue
        rows.append({
            "bucket": b,
            "count": count,
            "lambda_min": float(lam[sel].min()),
            "lambda_max": float(lam[sel].max()),
            "mse_mean": float(mse[sel].mean()),
            "mse_std": float(mse[sel].std()),
            "mae_mean": float(mae[sel].mean()),
            "mae_std": float(mae[sel].std()),
        })
    return rows


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
        "window", "k", "sep_mean", "sep_median", "auc_regime", "top10_in_reg1",
        "reg1_top10_coverage", "smooth_std_diff", "smooth_mean_abs_diff",
        "corr_mse", "corr_mae", "score"
    ]
    lines = [f"## {title}\n",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join([
            str(r.get("window", "")),
            str(r.get("k", "")),
            f"{float(r.get('sep_mean', np.nan)):.6f}",
            f"{float(r.get('sep_median', np.nan)):.6f}",
            f"{float(r.get('auc_regime', np.nan)):.6f}",
            f"{float(r.get('top10_in_reg1', np.nan)):.6f}",
            f"{float(r.get('reg1_top10_coverage', np.nan)):.6f}",
            f"{float(r.get('smooth_std_diff', np.nan)):.6f}",
            f"{float(r.get('smooth_mean_abs_diff', np.nan)):.6f}",
            f"{float(r.get('corr_mse', np.nan)):.6f}",
            f"{float(r.get('corr_mae', np.nan)):.6f}",
            f"{float(r.get('score', np.nan)):.6f}",
        ]) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def minmax_norm(values, nan_value=0.0):
    arr = np.array(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, 0.5)
    vmin = arr[finite].min()
    vmax = arr[finite].max()
    if vmax - vmin < 1e-12:
        out = np.full_like(arr, 0.5)
    else:
        out = (arr - vmin) / (vmax - vmin)
    out[~finite] = nan_value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--windows", type=str, default="20,30,50,80,120")
    parser.add_argument("--ks", type=str, default="2,3,5,8,10")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--t_switch", type=int, default=None)
    parser.add_argument("--n_buckets", type=int, default=5)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "exports_step4")
    os.makedirs(out_dir, exist_ok=True)

    x_path = find_x_npy(data_dir)
    X = np.load(x_path)
    if X.ndim != 2:
        raise ValueError(f"X.npy must be 2D (T,N), got shape {X.shape}")
    T, N = X.shape

    t_switch = read_t_switch(data_dir, args.t_switch)
    if t_switch is None:
        raise RuntimeError("t_switch not found. Provide --t_switch or ensure meta.json exists.")

    windows = parse_int_list(args.windows)
    ks = parse_int_list(args.ks)
    if not windows or not ks:
        raise RuntimeError("windows/ks list is empty.")

    # Ridge baseline (once)
    X_in = X[:-1]
    Y = X[1:]
    W = ridge_fit(X_in, Y, alpha=1.0)
    Y_pred = ridge_predict(X_in, W)
    err = Y_pred - Y
    mse_series = (err ** 2).mean(axis=1)
    mae_series = np.abs(err).mean(axis=1)

    results = []
    score_matrix = np.full((len(windows), len(ks)), np.nan, dtype=np.float64)

    for wi, window in enumerate(windows):
        feats, idx, valid_mask = build_window_features(X, window=window)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats_std = standardize_features(feats)

        for ki, k in enumerate(ks):
            k_use = int(k)
            if k_use <= 0:
                k_use = 1

            km_out = kmeans_sklearn(feats_std, k_use, args.seed)
            if km_out is None:
                _, centers = kmeans_simple(feats_std, k_use, args.seed, max_iter=args.max_iter)
            else:
                _, centers = km_out

            dists = nearest_center_distance(feats_std, centers)
            lambda_valid = quantile_normalize_p10_p90(dists)

            lambda_t = np.full((T,), np.nan, dtype=np.float64)
            lambda_t[idx] = lambda_valid

            # regime masks on valid positions
            t_valid = idx
            reg1_mask = t_valid >= t_switch
            reg0_mask = ~reg1_mask

            sep_mean = np.nan
            sep_median = np.nan
            if reg1_mask.any() and reg0_mask.any():
                sep_mean = float(lambda_valid[reg1_mask].mean() - lambda_valid[reg0_mask].mean())
                sep_median = float(np.median(lambda_valid[reg1_mask]) - np.median(lambda_valid[reg0_mask]))

            auc_regime = auc_roc(reg1_mask.astype(int), lambda_valid)

            n_valid = lambda_valid.size
            top_n = max(1, int(0.10 * n_valid)) if n_valid > 0 else 0
            top10_in_reg1 = np.nan
            reg1_top10_coverage = np.nan
            if top_n > 0:
                order = np.argsort(lambda_valid)[::-1]
                top_idx = order[:top_n]
                top_reg1 = int(reg1_mask[top_idx].sum())
                top10_in_reg1 = top_reg1 / float(top_n)
                n_reg1 = int(reg1_mask.sum())
                reg1_top10_coverage = (top_reg1 / float(n_reg1)) if n_reg1 > 0 else np.nan

            lam_valid_for_smooth = lambda_valid
            if lam_valid_for_smooth.size >= 2:
                diffs = np.diff(lam_valid_for_smooth)
                smooth_std_diff = float(diffs.std())
                smooth_mean_abs_diff = float(np.abs(diffs).mean())
            else:
                smooth_std_diff = np.nan
                smooth_mean_abs_diff = np.nan

            lam_corr = lambda_t[:-1]
            mask_corr = valid_mask[:-1] & np.isfinite(lam_corr)
            corr_mse = corrcoef_safe(lam_corr[mask_corr], mse_series[mask_corr])
            corr_mae = corrcoef_safe(lam_corr[mask_corr], mae_series[mask_corr])

            results.append({
                "window": int(window),
                "k": int(k_use),
                "sep_mean": sep_mean,
                "sep_median": sep_median,
                "auc_regime": auc_regime,
                "top10_in_reg1": top10_in_reg1,
                "reg1_top10_coverage": reg1_top10_coverage,
                "smooth_std_diff": smooth_std_diff,
                "smooth_mean_abs_diff": smooth_mean_abs_diff,
                "corr_mse": corr_mse,
                "corr_mae": corr_mae,
                "valid_count": int(valid_mask.sum()),
            })

    # score normalization and ranking
    aucs = [r["auc_regime"] for r in results]
    top10s = [r["top10_in_reg1"] for r in results]
    corr_mses = [r["corr_mse"] for r in results]
    sep_means = [r["sep_mean"] for r in results]
    smooth_stds = [r["smooth_std_diff"] for r in results]
    smooth_means = [r["smooth_mean_abs_diff"] for r in results]

    n_auc = minmax_norm(aucs, nan_value=0.0)
    n_top10 = minmax_norm(top10s, nan_value=0.0)
    n_corr = minmax_norm(corr_mses, nan_value=0.0)
    n_sep = minmax_norm(sep_means, nan_value=0.0)
    n_smooth_std = minmax_norm(smooth_stds, nan_value=1.0)
    n_smooth_mean = minmax_norm(smooth_means, nan_value=1.0)

    for i, r in enumerate(results):
        score = (
            n_auc[i] + n_top10[i] + n_corr[i] + n_sep[i] +
            (1.0 - n_smooth_std[i]) + (1.0 - n_smooth_mean[i])
        )
        r["score"] = float(score)
        r["score_norm_auc"] = float(n_auc[i])
        r["score_norm_top10"] = float(n_top10[i])
        r["score_norm_corr_mse"] = float(n_corr[i])
        r["score_norm_sep_mean"] = float(n_sep[i])
        r["score_norm_smooth_std"] = float(n_smooth_std[i])
        r["score_norm_smooth_mean_abs"] = float(n_smooth_mean[i])

    ranked = sorted(results, key=lambda x: (x["score"] if np.isfinite(x["score"]) else -1e9), reverse=True)

    # fill score heatmap
    score_lookup = {(r["window"], r["k"]): r["score"] for r in results}
    for wi, window in enumerate(windows):
        for ki, k in enumerate(ks):
            score_matrix[wi, ki] = score_lookup.get((window, k), np.nan)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_csv = os.path.join(out_dir, f"sweep_results_{ts}.csv")
    sweep_md = os.path.join(out_dir, f"sweep_results_{ts}.md")
    ranked_csv = os.path.join(out_dir, f"ranked_results_{ts}.csv")
    write_csv(results, sweep_csv)
    write_md(results, sweep_md, title="Sweep Results (window x k)")
    write_csv(ranked, ranked_csv)

    # best config outputs
    best = ranked[0]
    best_window = int(best["window"])
    best_k = int(best["k"])

    feats, idx, valid_mask = build_window_features(X, window=best_window)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats_std = standardize_features(feats)
    km_out = kmeans_sklearn(feats_std, best_k, args.seed)
    if km_out is None:
        _, centers = kmeans_simple(feats_std, best_k, args.seed, max_iter=args.max_iter)
    else:
        _, centers = km_out
    dists = nearest_center_distance(feats_std, centers)
    lambda_valid = quantile_normalize_p10_p90(dists)
    lambda_t = np.full((T,), np.nan, dtype=np.float64)
    lambda_t[idx] = lambda_valid

    best_cfg_path = os.path.join(out_dir, "best_config.json")
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    best_lambda_path = os.path.join(out_dir, "best_lambda_t.npy")
    np.save(best_lambda_path, lambda_t.astype(np.float32))

    fig_path = os.path.join(out_dir, "best_lambda_curve.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    raw_dist = np.full((T,), np.nan, dtype=np.float64)
    raw_dist[idx] = dists
    axes[0].plot(raw_dist, color="tab:gray", linewidth=1.0)
    axes[0].set_title("Raw distance to nearest KMeans center")
    axes[0].set_ylabel("distance")
    axes[1].plot(lambda_t, color="tab:blue", linewidth=1.0)
    axes[1].set_title("Lambda (p10/p90 normalized)")
    axes[1].set_ylabel("lambda_t")
    axes[1].set_xlabel("time t")
    axes[0].axvline(t_switch, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].axvline(t_switch, color="tab:red", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    lam_corr = lambda_t[:-1]
    mask_corr = valid_mask[:-1] & np.isfinite(lam_corr)
    lam_for_bucket = lam_corr[mask_corr]
    mse_for_bucket = mse_series[mask_corr]
    mae_for_bucket = mae_series[mask_corr]
    bucket_rows = compute_bucket_stats(lam_for_bucket, mse_for_bucket, mae_for_bucket, n_buckets=args.n_buckets)
    best_bucket_csv = os.path.join(out_dir, "best_bucket_stats.csv")
    write_csv(bucket_rows, best_bucket_csv)

    # heatmap score
    heatmap_path = os.path.join(out_dir, "heatmap_score.png")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(score_matrix, aspect="auto")
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(np.arange(len(windows)))
    ax.set_yticklabels([str(w) for w in windows])
    ax.set_xlabel("k")
    ax.set_ylabel("window")
    ax.set_title("Score heatmap")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    # pareto scatter: smooth_mean_abs_diff (x, lower is better) vs auc_regime (y, higher is better)
    pareto_path = os.path.join(out_dir, "pareto_scatter.png")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    x = np.array([r["smooth_mean_abs_diff"] for r in results], dtype=float)
    y = np.array([r["auc_regime"] for r in results], dtype=float)
    c = np.array([r["score"] for r in results], dtype=float)
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=40, edgecolors="none")
    ax.scatter([best["smooth_mean_abs_diff"]], [best["auc_regime"]], c="red", marker="*", s=120)
    ax.set_xlabel("smooth_mean_abs_diff (lower is better)")
    ax.set_ylabel("auc_regime (higher is better)")
    ax.set_title("Pareto scatter (AUC vs Smoothness)")
    fig.colorbar(sc, ax=ax, shrink=0.85, label="score")
    fig.tight_layout()
    fig.savefig(pareto_path, dpi=200)
    plt.close(fig)

    print("=== Step4: sweep window x k ===")
    print(f"X: {x_path} shape={X.shape}, t_switch={t_switch}")
    print(f"[OK] Saved: {sweep_csv}")
    print(f"[OK] Saved: {sweep_md}")
    print(f"[OK] Saved: {ranked_csv}")
    print(f"[OK] Saved: {best_cfg_path}")
    print(f"[OK] Saved: {best_lambda_path}")
    print(f"[OK] Saved: {fig_path}")
    print(f"[OK] Saved: {best_bucket_csv}")
    print(f"[OK] Saved: {heatmap_path}")
    print(f"[OK] Saved: {pareto_path}")
    print("Top-5 configs by score:")
    for i, r in enumerate(ranked[:5], start=1):
        print(f"{i:2d}. window={r['window']} k={r['k']} score={r['score']:.4f} "
              f"auc={r['auc_regime']:.4f} top10={r['top10_in_reg1']:.4f} corr_mse={r['corr_mse']:.4f} "
              f"sep_mean={r['sep_mean']:.4f} smooth={r['smooth_mean_abs_diff']:.4f}")


if __name__ == "__main__":
    main()
