import os
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


def skewness(x, eps=1e-8):
    # x: (W, N)
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
        return np.zeros((0, 4 * N), dtype=np.float64), np.array([], dtype=np.int64), valid_mask
    return np.vstack(feats).astype(np.float64), np.array(indices, dtype=np.int64), valid_mask


def standardize_features(F, eps=1e-8):
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (F - mu) / sd, mu, sd


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


def quantile_normalize(values):
    n = len(values)
    if n == 0:
        return values.copy()
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    denom = max(n - 1, 1)
    return ranks / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--t_switch", type=int, default=None)
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

    feats, idx, valid_mask = build_window_features(X, window=args.window)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats_std, mu, sd = standardize_features(feats)

    k = int(args.k)
    if k <= 0:
        k = 1

    km_out = kmeans_sklearn(feats_std, k, args.seed)
    if km_out is None:
        labels, centers = kmeans_simple(feats_std, k, args.seed, max_iter=args.max_iter)
        km_method = "simple"
    else:
        labels, centers = km_out
        km_method = "sklearn"

    dists = nearest_center_distance(feats_std, centers)
    lambda_valid = quantile_normalize(dists)

    lambda_t = np.full((T,), np.nan, dtype=np.float64)
    raw_dist = np.full((T,), np.nan, dtype=np.float64)
    lambda_t[idx] = lambda_valid
    raw_dist[idx] = dists

    lambda_t_path = os.path.join(out_dir, "lambda_t.npy")
    raw_dist_path = os.path.join(out_dir, "lambda_raw_dist.npy")
    valid_mask_path = os.path.join(out_dir, "lambda_valid_mask.npy")

    np.save(lambda_t_path, lambda_t.astype(np.float32))
    np.save(raw_dist_path, raw_dist.astype(np.float32))
    np.save(valid_mask_path, valid_mask.astype(np.bool_))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(out_dir, f"lambda_curve_{ts}.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(raw_dist, color="tab:gray", linewidth=1.0)
    axes[0].set_title("Raw distance to nearest KMeans center")
    axes[0].set_ylabel("distance")

    axes[1].plot(lambda_t, color="tab:blue", linewidth=1.0)
    axes[1].set_title("Lambda (quantile normalized)")
    axes[1].set_ylabel("lambda_t")
    axes[1].set_xlabel("time t")
    if t_switch is not None:
        for ax in axes:
            ax.axvline(t_switch, color="tab:red", linestyle="--", linewidth=1.0, label="t_switch")
            ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    def stats(arr):
        arr2 = arr[np.isfinite(arr)]
        if arr2.size == 0:
            return dict(count=0, mean=None, std=None, min=None, max=None)
        return dict(
            count=int(arr2.size),
            mean=float(arr2.mean()),
            std=float(arr2.std()),
            min=float(arr2.min()),
            max=float(arr2.max()),
        )

    summary = {
        "data_dir": data_dir,
        "x_path": x_path,
        "T": int(T),
        "N": int(N),
        "window": int(args.window),
        "k": int(k),
        "seed": int(args.seed),
        "kmeans_method": km_method,
        "valid_count": int(valid_mask.sum()),
        "t_switch": int(t_switch) if t_switch is not None else None,
        "raw_dist_stats": stats(raw_dist),
        "lambda_stats": stats(lambda_t),
        "outputs": {
            "lambda_t": lambda_t_path,
            "lambda_raw_dist": raw_dist_path,
            "lambda_valid_mask": valid_mask_path,
            "curve_fig": fig_path,
        },
    }

    summary_path = os.path.join(out_dir, "lambda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Step4: build lambda (kmeans) ===")
    print(f"X: {x_path} shape={X.shape}")
    print(f"window={args.window}, valid={int(valid_mask.sum())}/{T}, k={k}, kmeans={km_method}")
    if t_switch is not None:
        print(f"t_switch={t_switch}")
    print(f"raw_dist stats: {summary['raw_dist_stats']}")
    print(f"lambda stats:   {summary['lambda_stats']}")
    print(f"[OK] Saved: {lambda_t_path}")
    print(f"[OK] Saved: {raw_dist_path}")
    print(f"[OK] Saved: {valid_mask_path}")
    print(f"[OK] Saved: {fig_path}")
    print(f"[OK] Saved: {summary_path}")


if __name__ == "__main__":
    main()
