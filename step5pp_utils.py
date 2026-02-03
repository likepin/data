import os
import numpy as np


def skewness(x, eps=1e-8):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    centered = x - mean
    m3 = (centered ** 3).mean(axis=0)
    return m3 / (std ** 3 + eps)


def build_window_features(X, window):
    T, N = X.shape
    feats = []
    indices = []
    valid_mask = np.zeros(T, dtype=bool)
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


def compute_lambda_kmeans(X, window, k, seed=123, max_iter=100):
    feats, idx, valid_mask = build_window_features(X, window=window)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats_std = standardize_features(feats)
    km_out = kmeans_sklearn(feats_std, k, seed)
    if km_out is None:
        _, centers = kmeans_simple(feats_std, k, seed, max_iter=max_iter)
    else:
        _, centers = km_out
    dists = nearest_center_distance(feats_std, centers)
    lambda_valid = quantile_normalize_p10_p90(dists)
    lambda_t = np.full((X.shape[0],), np.nan, dtype=np.float64)
    lambda_t[idx] = lambda_valid
    return lambda_t, valid_mask


def pick_lambda_configs_from_step4(data_dir, score_type, top_m):
    out_dir = os.path.join(data_dir, "exports_step4")
    if not os.path.isdir(out_dir):
        raise FileNotFoundError("exports_step4 not found.")
    candidates = [f for f in os.listdir(out_dir) if f.startswith("rescored_results_") and f.endswith(".csv")]
    if not candidates:
        raise FileNotFoundError("rescored_results_*.csv not found in exports_step4.")
    path = os.path.join(out_dir, sorted(candidates)[-1])
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            r = {header[i]: vals[i] if i < len(vals) else "" for i in range(len(header))}
            rows.append(r)
    key = score_type
    def to_float(v):
        try:
            return float(v)
        except Exception:
            return -1e9
    rows_sorted = sorted(rows, key=lambda r: to_float(r.get(key, "")), reverse=True)
    out = []
    for r in rows_sorted[:top_m]:
        out.append({
            "window": int(float(r.get("window", 0))),
            "k": int(float(r.get("k", 0))),
            "score": to_float(r.get(key, "")),
        })
    return out
