import os
import csv
import json
import argparse
from datetime import datetime

import numpy as np


def find_x_npy(data_dir):
    direct = os.path.join(data_dir, "X.npy")
    if os.path.isfile(direct):
        return direct
    for root, _, files in os.walk(data_dir):
        if "X.npy" in files:
            return os.path.join(root, "X.npy")
    raise FileNotFoundError(f"X.npy not found under: {data_dir}")


def find_lambda_files(data_dir, out_dir):
    candidates = [
        os.path.join(out_dir, "lambda_t.npy"),
        os.path.join(data_dir, "lambda_t.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    newest = None
    newest_mtime = -1
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().startswith("lambda_t") and f.lower().endswith(".npy"):
                p = os.path.join(root, f)
                m = os.path.getmtime(p)
                if m > newest_mtime:
                    newest_mtime = m
                    newest = p
    if newest is None:
        raise FileNotFoundError(f"lambda_t.npy not found under: {data_dir}")
    return newest


def find_valid_mask(data_dir, out_dir):
    candidates = [
        os.path.join(out_dir, "lambda_valid_mask.npy"),
        os.path.join(data_dir, "lambda_valid_mask.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def ridge_fit(X, Y, alpha=1.0, fit_intercept=True):
    # X: (n, d), Y: (n, m)
    if fit_intercept:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_aug = np.concatenate([X, ones], axis=1)
    else:
        X_aug = X
    d = X_aug.shape[1]
    XtX = X_aug.T @ X_aug
    reg = alpha * np.eye(d, dtype=X_aug.dtype)
    XtY = X_aug.T @ Y
    W = np.linalg.solve(XtX + reg, XtY)
    return W


def ridge_predict(X, W, fit_intercept=True):
    if fit_intercept:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_aug = np.concatenate([X, ones], axis=1)
    else:
        X_aug = X
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


def write_md(rows, out_path, title="Lambda Bucket Difficulty"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = ["bucket", "count", "lambda_min", "lambda_max", "mse_mean", "mse_std", "mae_mean", "mae_std"]
    lines = [f"## {title}\n",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join([
            str(r.get("bucket", "")),
            str(r.get("count", "")),
            f"{float(r.get('lambda_min', 0)):.4f}",
            f"{float(r.get('lambda_max', 0)):.4f}",
            f"{float(r.get('mse_mean', 0)):.6f}",
            f"{float(r.get('mse_std', 0)):.6f}",
            f"{float(r.get('mae_mean', 0)):.6f}",
            f"{float(r.get('mae_std', 0)):.6f}",
        ]) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--no_intercept", action="store_true")
    parser.add_argument("--n_buckets", type=int, default=5)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or os.path.join(data_dir, "exports_step4")
    os.makedirs(out_dir, exist_ok=True)

    x_path = find_x_npy(data_dir)
    X = np.load(x_path)
    if X.ndim != 2:
        raise ValueError(f"X.npy must be 2D (T,N), got shape {X.shape}")

    lambda_path = find_lambda_files(data_dir, out_dir)
    lambda_t = np.load(lambda_path)
    if lambda_t.ndim != 1:
        lambda_t = lambda_t.reshape(-1)

    valid_mask_path = find_valid_mask(data_dir, out_dir)
    if valid_mask_path is not None:
        valid_mask = np.load(valid_mask_path).astype(bool)
    else:
        valid_mask = np.isfinite(lambda_t)

    T, N = X.shape
    if lambda_t.shape[0] != T:
        raise ValueError(f"lambda_t length {lambda_t.shape[0]} does not match X length {T}")

    X_in = X[:-1]
    Y = X[1:]
    fit_intercept = not args.no_intercept
    W = ridge_fit(X_in, Y, alpha=args.alpha, fit_intercept=fit_intercept)
    Y_pred = ridge_predict(X_in, W, fit_intercept=fit_intercept)

    err = Y_pred - Y
    mse = (err ** 2).mean(axis=1)
    mae = np.abs(err).mean(axis=1)

    lam = lambda_t[:-1]
    mask = valid_mask[:-1] & np.isfinite(lam)
    lam = lam[mask]
    mse = mse[mask]
    mae = mae[mask]

    if lam.size == 0:
        raise RuntimeError("No valid lambda values after masking.")

    edges = make_bucket_edges(lam, args.n_buckets)
    bidx = bucket_assign(lam, edges)

    rows = []
    for b in range(args.n_buckets):
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

    # global correlation (Pearson)
    corr_mse = float(np.corrcoef(lam, mse)[0, 1])
    corr_mae = float(np.corrcoef(lam, mae)[0, 1])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"lambda_bucket_stats_{ts}.csv")
    out_md = os.path.join(out_dir, f"lambda_bucket_stats_{ts}.md")
    write_csv(rows, out_csv)
    write_md(rows, out_md)

    summary = {
        "data_dir": data_dir,
        "x_path": x_path,
        "lambda_path": lambda_path,
        "T": int(T),
        "N": int(N),
        "alpha": float(args.alpha),
        "fit_intercept": bool(fit_intercept),
        "n_buckets": int(args.n_buckets),
        "corr_lambda_mse": corr_mse,
        "corr_lambda_mae": corr_mae,
        "outputs": {"csv": out_csv, "md": out_md},
    }
    summary_path = os.path.join(out_dir, f"lambda_bucket_summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Step4: bucket difficulty (ridge baseline) ===")
    print(f"X: {x_path} shape={X.shape}")
    print(f"lambda: {lambda_path}, valid={int(mask.sum())}/{T-1}")
    print(f"alpha={args.alpha}, fit_intercept={fit_intercept}")
    print(f"corr(lambda, mse)={corr_mse:.4f}, corr(lambda, mae)={corr_mae:.4f}")
    print(f"[OK] Saved: {out_csv}")
    print(f"[OK] Saved: {out_md}")
    print(f"[OK] Saved: {summary_path}")


if __name__ == "__main__":
    main()
