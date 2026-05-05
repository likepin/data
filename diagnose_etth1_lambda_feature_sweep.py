from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posthoc_ecl96_deltaA_manual_gate import find_result_dirs
from real_dataset_io import load_values_matrix


DATA_CSV = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\dataset\ETTh1.csv")
RESULT_ROOT = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\results")
OUT_DIR = Path(r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\etth196_lambda_feature_sweep")
BASELINE_PATTERN = "etth196_validate_baseline_itr3_*projection_*"
OUTPUT_PREFIX = "etth196"
DATA_DATE_COL = "date"
DATA_HEADER_MODE = "infer"
DATA_SEP = ","

WINDOWS = [20, 40, 60, 80, 120]
KS = [2, 3, 5, 8]
TOP_TEST_CONFIGS = 20
CALIBRATION_FRACTION = 0.5
VALIDATION_FOLDS = 4
MODES = [
    "current",
    "change_half",
    "change_slope",
    "change_slope_no_range",
    "level_shift",
    "volatility",
    "tail_risk",
]
SPEARMAN_FLOOR = 0.15
BUCKET5_LIFT_FLOOR = 5.0
BUCKET_TREND_FLOOR = 0.0
STABLE_SPEARMAN_MEAN_FLOOR = 0.10
STABLE_SPEARMAN_MIN_FLOOR = -0.05
STABLE_BUCKET5_POSITIVE_FRACTION = 0.75
SEED = 2023
SEQ_LEN = 96
PRED_LEN = 96
TRAIN_END = 12 * 30 * 24
VAL_END = TRAIN_END + 4 * 30 * 24
TEST_END = TRAIN_END + 8 * 30 * 24


def skewness(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    centered = x - mean
    m3 = (centered**3).mean(axis=0)
    return m3 / (std**3 + eps)


def kurtosis(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    centered = x - mean
    m4 = (centered**4).mean(axis=0)
    return m4 / (std**4 + eps)


def slope_feature(w: np.ndarray) -> np.ndarray:
    t = np.arange(w.shape[0], dtype=np.float64)
    t = t - t.mean()
    denom = np.sum(t * t)
    if denom <= 0:
        return np.zeros((w.shape[1],), dtype=np.float64)
    centered = w - w.mean(axis=0, keepdims=True)
    return (t[:, None] * centered).sum(axis=0) / denom


def build_window_features(X: np.ndarray, window: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    indices = []
    for t in range(window - 1, X.shape[0]):
        w = X[t - window + 1 : t + 1]
        std = w.std(axis=0)
        rng = w.max(axis=0) - w.min(axis=0)
        q25, q50, q75, q95, q99 = np.quantile(w, [0.25, 0.50, 0.75, 0.95, 0.99], axis=0)
        iqr = q75 - q25
        tail_95 = q95 - q50
        tail_99 = q99 - q50
        split = max(1, w.shape[0] // 2)
        half_mean_delta = w[-split:].mean(axis=0) - w[:split].mean(axis=0)
        diff = np.diff(w, axis=0)
        diff_mean = diff.mean(axis=0) if diff.shape[0] else np.zeros(X.shape[1], dtype=np.float64)
        diff_std = diff.std(axis=0) if diff.shape[0] else np.zeros(X.shape[1], dtype=np.float64)
        diff_range = diff.max(axis=0) - diff.min(axis=0) if diff.shape[0] else np.zeros(X.shape[1], dtype=np.float64)
        slope = slope_feature(w)
        if mode == "current":
            feat = np.concatenate([w.mean(axis=0), std, skewness(w), rng, half_mean_delta], axis=0)
        elif mode == "change_half":
            feat = np.concatenate([std, rng, half_mean_delta, diff_mean, diff_std], axis=0)
        elif mode == "change_slope":
            feat = np.concatenate([std, rng, slope, diff_mean, diff_std], axis=0)
        elif mode == "change_slope_no_range":
            feat = np.concatenate([std, slope, diff_mean, diff_std], axis=0)
        elif mode == "level_shift":
            feat = np.concatenate([w.mean(axis=0), half_mean_delta, slope, diff_mean, std], axis=0)
        elif mode == "volatility":
            feat = np.concatenate([std, rng, iqr, diff_std, diff_range], axis=0)
        elif mode == "tail_risk":
            feat = np.concatenate([std, iqr, tail_95, tail_99, kurtosis(w)], axis=0)
        else:
            raise ValueError(f"Unknown feature mode: {mode}")
        feats.append(feat)
        indices.append(t)
    return np.asarray(feats, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def kmeans_sklearn(F: np.ndarray, k: int, seed: int):
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return None
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(F)
    return labels, model.cluster_centers_


def kmeans_simple(F: np.ndarray, k: int, seed: int, max_iter: int = 100, tol: float = 1e-4):
    rng = np.random.RandomState(seed)
    k = min(k, F.shape[0])
    centers = F[rng.choice(F.shape[0], size=k, replace=False)].copy()
    labels = np.zeros((F.shape[0],), dtype=np.int64)
    for _ in range(max_iter):
        d2 = ((F[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = new_labels == j
            new_centers[j] = F[mask].mean(axis=0) if mask.any() else F[rng.randint(0, F.shape[0])]
        shift = np.sqrt(((centers - new_centers) ** 2).sum(axis=1)).mean()
        centers = new_centers
        labels = new_labels
        if shift < tol:
            break
    return labels, centers


def nearest_center_distance(F: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = ((F[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def normalize_by_train_quantiles(values: np.ndarray, train_values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q10 = np.quantile(train_values, 0.10)
    q90 = np.quantile(train_values, 0.90)
    if not np.isfinite(q10) or not np.isfinite(q90) or q90 <= q10 + eps:
        vmin = float(train_values.min())
        vmax = float(train_values.max())
        if vmax <= vmin + eps:
            return np.zeros_like(values)
        return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.clip((values - q10) / (q90 - q10), 0.0, 1.0)


def sanitize_lambda(lambda_t: np.ndarray) -> np.ndarray:
    if np.isinf(lambda_t).any():
        raise ValueError("lambda contains Inf")
    nan_mask = np.isnan(lambda_t)
    if not nan_mask.any():
        return lambda_t.astype(np.float32)
    valid = np.where(~nan_mask)[0]
    if valid.size == 0:
        raise ValueError("lambda entirely NaN")
    return np.interp(np.arange(lambda_t.shape[0]), valid, lambda_t[valid]).astype(np.float32)


def compute_lambda_timeline(full_z: np.ndarray, window: int, k: int, mode: str) -> np.ndarray:
    train_z = full_z[:TRAIN_END]
    train_feats, _ = build_window_features(train_z, window=window, mode=mode)
    full_feats, full_idx = build_window_features(full_z, window=window, mode=mode)
    train_feats = np.nan_to_num(train_feats, nan=0.0, posinf=0.0, neginf=0.0)
    full_feats = np.nan_to_num(full_feats, nan=0.0, posinf=0.0, neginf=0.0)
    feat_mean = train_feats.mean(axis=0)
    feat_std = train_feats.std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    train_std = (train_feats - feat_mean) / feat_std
    full_std = (full_feats - feat_mean) / feat_std
    km = kmeans_sklearn(train_std, k=k, seed=SEED)
    if km is None:
        _, centers = kmeans_simple(train_std, k=k, seed=SEED)
    else:
        _, centers = km
    train_dist = nearest_center_distance(train_std, centers)
    full_dist = nearest_center_distance(full_std, centers)
    lambda_valid = normalize_by_train_quantiles(full_dist, train_dist)
    lambda_t = np.full((full_z.shape[0],), np.nan, dtype=np.float64)
    lambda_t[full_idx] = lambda_valid
    return sanitize_lambda(lambda_t)


def lambda_for_split(lambda_t: np.ndarray, split: str) -> np.ndarray:
    if split == "val":
        border1, border2 = TRAIN_END - SEQ_LEN, VAL_END
    elif split == "test":
        border1, border2 = VAL_END - SEQ_LEN, TEST_END
    else:
        raise ValueError(split)
    n = border2 - border1 - SEQ_LEN - PRED_LEN + 1
    values = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        s = border1 + i
        values[i] = float(lambda_t[s : s + SEQ_LEN].mean())
    return values


def load_full_z() -> np.ndarray:
    values, _columns = load_values_matrix(
        DATA_CSV,
        date_col=DATA_DATE_COL,
        value_cols=None,
        header_mode=DATA_HEADER_MODE,
        sep=DATA_SEP,
    )
    train = values[:TRAIN_END]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def load_baseline_sample_errors(split: str) -> tuple[np.ndarray, np.ndarray]:
    pred_name = "val_pred.npy" if split == "val" else "pred.npy"
    true_name = "val_true.npy" if split == "val" else "true.npy"
    dirs = find_result_dirs(RESULT_ROOT, BASELINE_PATTERN, pred_file=pred_name, true_file=true_name)
    mse_rows = []
    mae_rows = []
    for d in dirs:
        pred = np.load(d / pred_name, mmap_mode="r")
        true = np.load(d / true_name, mmap_mode="r")
        err = np.asarray(true, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        mse_rows.append(np.mean(err * err, axis=(1, 2)))
        mae_rows.append(np.mean(np.abs(err), axis=(1, 2)))
    return np.stack(mse_rows).mean(axis=0), np.stack(mae_rows).mean(axis=0)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    return float(np.sum(rx * ry) / denom) if denom > 0 else 0.0


def bucket_summary(lambda_values: np.ndarray, mse: np.ndarray, mae: np.ndarray) -> tuple[pd.DataFrame, dict]:
    order = np.argsort(lambda_values, kind="mergesort")
    rows = []
    for bucket, idx in enumerate(np.array_split(order, 5), start=1):
        idx = np.asarray(idx, dtype=np.int64)
        rows.append(
            {
                "bucket": bucket,
                "n": int(idx.size),
                "lambda_min": float(lambda_values[idx].min()),
                "lambda_mean": float(lambda_values[idx].mean()),
                "lambda_max": float(lambda_values[idx].max()),
                "mse": float(mse[idx].mean()),
                "mae": float(mae[idx].mean()),
            }
        )
    df = pd.DataFrame(rows)
    trend = spearman(df["bucket"].to_numpy(dtype=np.float64), df["mse"].to_numpy(dtype=np.float64))
    global_mse = float(np.mean(mse))
    global_mae = float(np.mean(mae))
    bucket5_mse = float(df.loc[df["bucket"] == 5, "mse"].iloc[0])
    bucket5_mae = float(df.loc[df["bucket"] == 5, "mae"].iloc[0])
    metrics = {
        "bucket_mse_spearman": trend,
        "bucket5_mse_lift_pct": 100.0 * (bucket5_mse - global_mse) / global_mse,
        "bucket5_mae_lift_pct": 100.0 * (bucket5_mae - global_mae) / global_mae,
        "bucket5_vs_bucket1_mse_pct": 100.0
        * (bucket5_mse - float(df.loc[df["bucket"] == 1, "mse"].iloc[0]))
        / float(df.loc[df["bucket"] == 1, "mse"].iloc[0]),
    }
    return df, metrics


def evaluate_config(full_z: np.ndarray, val_mse: np.ndarray, val_mae: np.ndarray, mode: str, window: int, k: int):
    lambda_t = compute_lambda_timeline(full_z, window=window, k=k, mode=mode)
    lambda_val = lambda_for_split(lambda_t, "val")
    bucket_df, bucket_metrics = bucket_summary(lambda_val, val_mse, val_mae)
    row = {
        "mode": mode,
        "window": window,
        "k": k,
        "lambda_mean": float(lambda_val.mean()),
        "lambda_std": float(lambda_val.std()),
        "lambda_iqr": float(np.quantile(lambda_val, 0.75) - np.quantile(lambda_val, 0.25)),
        "lambda_min": float(lambda_val.min()),
        "lambda_max": float(lambda_val.max()),
        "spearman_mse": spearman(lambda_val, val_mse),
        "spearman_mae": spearman(lambda_val, val_mae),
        **bucket_metrics,
    }
    bucket_df.insert(0, "k", k)
    bucket_df.insert(0, "window", window)
    bucket_df.insert(0, "mode", mode)
    return row, bucket_df, lambda_t


def summarize_lambda_errors(lambda_values: np.ndarray, mse: np.ndarray, mae: np.ndarray) -> tuple[dict, pd.DataFrame]:
    bucket_df, bucket_metrics = bucket_summary(lambda_values, mse, mae)
    row = {
        "lambda_mean": float(lambda_values.mean()),
        "lambda_std": float(lambda_values.std()),
        "lambda_iqr": float(np.quantile(lambda_values, 0.75) - np.quantile(lambda_values, 0.25)),
        "lambda_min": float(lambda_values.min()),
        "lambda_max": float(lambda_values.max()),
        "spearman_mse": spearman(lambda_values, mse),
        "spearman_mae": spearman(lambda_values, mae),
        **bucket_metrics,
    }
    return row, bucket_df


def add_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["passes_spearman"] = df["spearman_mse"] >= SPEARMAN_FLOOR
    df["passes_bucket5"] = df["bucket5_mse_lift_pct"] >= BUCKET5_LIFT_FLOOR
    df["passes_trend"] = df["bucket_mse_spearman"] > BUCKET_TREND_FLOOR
    df["gate_candidate"] = df["passes_spearman"] & df["passes_bucket5"] & df["passes_trend"]
    bucket5_score = np.clip(df["bucket5_mse_lift_pct"].to_numpy(dtype=np.float64) / 20.0, -1.0, 1.0)
    df["selection_score"] = (
        0.40 * df["spearman_mse"].to_numpy(dtype=np.float64)
        + 0.40 * bucket5_score
        + 0.20 * df["bucket_mse_spearman"].to_numpy(dtype=np.float64)
    )
    return df


def build_fold_stability(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, window, k), sub in fold_df.groupby(["mode", "window", "k"], sort=False):
        spearman_values = sub["spearman_mse"].to_numpy(dtype=np.float64)
        bucket5_values = sub["bucket5_mse_lift_pct"].to_numpy(dtype=np.float64)
        trend_values = sub["bucket_mse_spearman"].to_numpy(dtype=np.float64)
        positive_bucket5_fraction = float(np.mean(bucket5_values > 0.0))
        positive_spearman_fraction = float(np.mean(spearman_values > 0.0))
        stable_candidate = (
            spearman_values.mean() >= STABLE_SPEARMAN_MEAN_FLOOR
            and spearman_values.min() >= STABLE_SPEARMAN_MIN_FLOOR
            and positive_bucket5_fraction >= STABLE_BUCKET5_POSITIVE_FRACTION
            and trend_values.mean() > 0.0
        )
        stability_score = (
            0.35 * spearman_values.mean()
            - 0.20 * spearman_values.std()
            + 0.25 * np.clip(bucket5_values.mean() / 20.0, -1.0, 1.0)
            + 0.15 * trend_values.mean()
            + 0.05 * positive_bucket5_fraction
        )
        rows.append(
            {
                "mode": mode,
                "window": int(window),
                "k": int(k),
                "fold_spearman_mean": float(spearman_values.mean()),
                "fold_spearman_std": float(spearman_values.std()),
                "fold_spearman_min": float(spearman_values.min()),
                "fold_spearman_max": float(spearman_values.max()),
                "fold_bucket5_lift_mean": float(bucket5_values.mean()),
                "fold_bucket5_lift_min": float(bucket5_values.min()),
                "fold_bucket5_lift_max": float(bucket5_values.max()),
                "fold_trend_mean": float(trend_values.mean()),
                "fold_trend_min": float(trend_values.min()),
                "positive_spearman_fraction": positive_spearman_fraction,
                "positive_bucket5_fraction": positive_bucket5_fraction,
                "stable_candidate": stable_candidate,
                "stability_score": float(stability_score),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_candidate", "stability_score", "fold_spearman_mean", "fold_bucket5_lift_mean"],
        ascending=[False, False, False, False],
    )


def plot_top_bucket_trends(bucket_df: pd.DataFrame, top_configs: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, cfg in top_configs.iterrows():
        mask = (
            (bucket_df["mode"] == cfg["mode"])
            & (bucket_df["window"] == cfg["window"])
            & (bucket_df["k"] == cfg["k"])
        )
        sub = bucket_df.loc[mask]
        label = f"{cfg['mode']} w={int(cfg['window'])} k={int(cfg['k'])}"
        ax.plot(sub["bucket"], sub["mse"], marker="o", label=label)
    ax.set_title(f"{OUTPUT_PREFIX} Val Sample MSE by Lambda Bucket")
    ax.set_xlabel("lambda bucket")
    ax.set_ylabel("baseline sample MSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_z = load_full_z()
    val_mse, val_mae = load_baseline_sample_errors("val")
    test_mse, test_mae = load_baseline_sample_errors("test")

    rows = []
    bucket_frames = []
    lambda_cache: dict[tuple[str, int, int], np.ndarray] = {}
    for mode in MODES:
        for window in WINDOWS:
            for k in KS:
                row, bucket_df, lambda_t = evaluate_config(full_z, val_mse, val_mae, mode, window, k)
                rows.append(row)
                bucket_frames.append(bucket_df)
                lambda_cache[(mode, window, k)] = lambda_t
                print(
                    f"[Val] mode={mode} window={window} k={k} "
                    f"rho_mse={row['spearman_mse']:.4f} "
                    f"bucket5_lift={row['bucket5_mse_lift_pct']:.2f}%",
                    flush=True,
                )

    val_df = add_selection_columns(pd.DataFrame(rows))
    val_df = val_df.sort_values(
        ["gate_candidate", "selection_score", "spearman_mse", "bucket5_mse_lift_pct"],
        ascending=[False, False, False, False],
    )
    bucket_df_all = pd.concat(bucket_frames, ignore_index=True)
    val_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_val.csv"
    bucket_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_val_buckets.csv"
    val_df.to_csv(val_path, index=False)
    bucket_df_all.to_csv(bucket_path, index=False)

    cal_n = max(1, int(len(val_mse) * CALIBRATION_FRACTION))
    calibration_rows = []
    for (mode, window, k), lambda_t in lambda_cache.items():
        lambda_val = lambda_for_split(lambda_t, "val")
        cal_metrics, _ = summarize_lambda_errors(lambda_val[:cal_n], val_mse[:cal_n], val_mae[:cal_n])
        calibration_rows.append({"mode": mode, "window": window, "k": k, **cal_metrics})

    calibration_df = add_selection_columns(pd.DataFrame(calibration_rows))
    calibration_df = calibration_df.sort_values(
        ["gate_candidate", "selection_score", "spearman_mse", "bucket5_mse_lift_pct"],
        ascending=[False, False, False, False],
    )
    calibration_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_calibration.csv"
    calibration_df.to_csv(calibration_path, index=False)

    holdout_rows = []
    holdout_buckets = []
    for _, cfg in calibration_df.head(TOP_TEST_CONFIGS).iterrows():
        cfg_key = (str(cfg["mode"]), int(cfg["window"]), int(cfg["k"]))
        cfg_lambda_val = lambda_for_split(lambda_cache[cfg_key], "val")
        hold_metrics, hold_bucket_df = summarize_lambda_errors(
            cfg_lambda_val[cal_n:], val_mse[cal_n:], val_mae[cal_n:]
        )
        holdout_rows.append(
            {
                "mode": cfg_key[0],
                "window": cfg_key[1],
                "k": cfg_key[2],
                "calibration_selection_score": float(cfg["selection_score"]),
                "calibration_spearman_mse": float(cfg["spearman_mse"]),
                "calibration_bucket5_mse_lift_pct": float(cfg["bucket5_mse_lift_pct"]),
                **hold_metrics,
            }
        )
        hold_bucket_df.insert(0, "k", cfg_key[2])
        hold_bucket_df.insert(0, "window", cfg_key[1])
        hold_bucket_df.insert(0, "mode", cfg_key[0])
        holdout_buckets.append(hold_bucket_df)

    holdout_df = add_selection_columns(pd.DataFrame(holdout_rows))
    holdout_df = holdout_df.sort_values(
        ["gate_candidate", "selection_score", "spearman_mse", "bucket5_mse_lift_pct"],
        ascending=[False, False, False, False],
    )
    holdout_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_calibration_top_holdout.csv"
    holdout_bucket_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_calibration_top_holdout_buckets.csv"
    holdout_df.to_csv(holdout_path, index=False)
    pd.concat(holdout_buckets, ignore_index=True).to_csv(holdout_bucket_path, index=False)

    holdout_best = holdout_df.iloc[0]
    holdout_key = (str(holdout_best["mode"]), int(holdout_best["window"]), int(holdout_best["k"]))
    holdout_lambda_test = lambda_for_split(lambda_cache[holdout_key], "test")
    holdout_test_metrics, holdout_test_bucket_df = summarize_lambda_errors(holdout_lambda_test, test_mse, test_mae)
    holdout_test_row = {
        "mode": holdout_key[0],
        "window": holdout_key[1],
        "k": holdout_key[2],
        "calibration_selection_score": float(holdout_best["calibration_selection_score"]),
        "calibration_spearman_mse": float(holdout_best["calibration_spearman_mse"]),
        "calibration_bucket5_mse_lift_pct": float(holdout_best["calibration_bucket5_mse_lift_pct"]),
        "holdout_selection_score": float(holdout_best["selection_score"]),
        "holdout_spearman_mse": float(holdout_best["spearman_mse"]),
        "holdout_bucket5_mse_lift_pct": float(holdout_best["bucket5_mse_lift_pct"]),
        **holdout_test_metrics,
    }
    holdout_test_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_holdout_selected_test.csv"
    holdout_test_bucket_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_holdout_selected_test_buckets.csv"
    pd.DataFrame([holdout_test_row]).to_csv(holdout_test_path, index=False)
    holdout_test_bucket_df.to_csv(holdout_test_bucket_path, index=False)

    fold_rows = []
    fold_indices = np.array_split(np.arange(len(val_mse)), VALIDATION_FOLDS)
    for (mode, window, k), lambda_t in lambda_cache.items():
        lambda_val = lambda_for_split(lambda_t, "val")
        for fold_id, idx in enumerate(fold_indices, start=1):
            fold_metrics, _ = summarize_lambda_errors(lambda_val[idx], val_mse[idx], val_mae[idx])
            fold_rows.append(
                {
                    "mode": mode,
                    "window": window,
                    "k": k,
                    "fold": fold_id,
                    "fold_start": int(idx[0]),
                    "fold_end": int(idx[-1]),
                    **fold_metrics,
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    stability_df = build_fold_stability(fold_df)
    fold_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_validation_folds.csv"
    stability_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_validation_fold_stability.csv"
    fold_df.to_csv(fold_path, index=False)
    stability_df.to_csv(stability_path, index=False)

    stable_best = stability_df.iloc[0]
    stable_key = (str(stable_best["mode"]), int(stable_best["window"]), int(stable_best["k"]))
    stable_lambda_test = lambda_for_split(lambda_cache[stable_key], "test")
    stable_test_metrics, stable_test_bucket_df = summarize_lambda_errors(stable_lambda_test, test_mse, test_mae)
    stable_test_row = {
        "mode": stable_key[0],
        "window": stable_key[1],
        "k": stable_key[2],
        "stable_candidate": bool(stable_best["stable_candidate"]),
        "stability_score": float(stable_best["stability_score"]),
        "fold_spearman_mean": float(stable_best["fold_spearman_mean"]),
        "fold_spearman_min": float(stable_best["fold_spearman_min"]),
        "fold_bucket5_lift_mean": float(stable_best["fold_bucket5_lift_mean"]),
        "fold_bucket5_lift_min": float(stable_best["fold_bucket5_lift_min"]),
        "positive_bucket5_fraction": float(stable_best["positive_bucket5_fraction"]),
        **stable_test_metrics,
    }
    stable_test_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_fold_stable_selected_test.csv"
    stable_test_bucket_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_fold_stable_selected_test_buckets.csv"
    pd.DataFrame([stable_test_row]).to_csv(stable_test_path, index=False)
    stable_test_bucket_df.to_csv(stable_test_bucket_path, index=False)

    best = val_df.iloc[0]
    key = (str(best["mode"]), int(best["window"]), int(best["k"]))
    lambda_test = lambda_for_split(lambda_cache[key], "test")
    test_bucket_df, test_bucket_metrics = bucket_summary(lambda_test, test_mse, test_mae)
    test_row = {
        "mode": key[0],
        "window": key[1],
        "k": key[2],
        "lambda_mean": float(lambda_test.mean()),
        "lambda_std": float(lambda_test.std()),
        "lambda_iqr": float(np.quantile(lambda_test, 0.75) - np.quantile(lambda_test, 0.25)),
        "lambda_min": float(lambda_test.min()),
        "lambda_max": float(lambda_test.max()),
        "spearman_mse": spearman(lambda_test, test_mse),
        "spearman_mae": spearman(lambda_test, test_mae),
        **test_bucket_metrics,
    }
    test_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_test_selected.csv"
    test_bucket_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_test_selected_buckets.csv"
    pd.DataFrame([test_row]).to_csv(test_path, index=False)
    test_bucket_df.to_csv(test_bucket_path, index=False)

    top_test_rows = []
    top_test_buckets = []
    for _, cfg in val_df.head(TOP_TEST_CONFIGS).iterrows():
        cfg_key = (str(cfg["mode"]), int(cfg["window"]), int(cfg["k"]))
        cfg_lambda_test = lambda_for_split(lambda_cache[cfg_key], "test")
        cfg_bucket_df, cfg_bucket_metrics = bucket_summary(cfg_lambda_test, test_mse, test_mae)
        cfg_test_row = {
            "mode": cfg_key[0],
            "window": cfg_key[1],
            "k": cfg_key[2],
            "val_selection_score": float(cfg["selection_score"]),
            "val_spearman_mse": float(cfg["spearman_mse"]),
            "val_bucket5_mse_lift_pct": float(cfg["bucket5_mse_lift_pct"]),
            "lambda_mean": float(cfg_lambda_test.mean()),
            "lambda_std": float(cfg_lambda_test.std()),
            "lambda_iqr": float(np.quantile(cfg_lambda_test, 0.75) - np.quantile(cfg_lambda_test, 0.25)),
            "lambda_min": float(cfg_lambda_test.min()),
            "lambda_max": float(cfg_lambda_test.max()),
            "spearman_mse": spearman(cfg_lambda_test, test_mse),
            "spearman_mae": spearman(cfg_lambda_test, test_mae),
            **cfg_bucket_metrics,
        }
        top_test_rows.append(cfg_test_row)
        cfg_bucket_df.insert(0, "k", cfg_key[2])
        cfg_bucket_df.insert(0, "window", cfg_key[1])
        cfg_bucket_df.insert(0, "mode", cfg_key[0])
        top_test_buckets.append(cfg_bucket_df)

    top_test_df = pd.DataFrame(top_test_rows)
    top_test_df = top_test_df.sort_values(
        ["spearman_mse", "bucket5_mse_lift_pct", "bucket_mse_spearman"],
        ascending=[False, False, False],
    )
    top_test_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_test_top_val.csv"
    top_test_bucket_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_test_top_val_buckets.csv"
    top_test_df.to_csv(top_test_path, index=False)
    pd.concat(top_test_buckets, ignore_index=True).to_csv(top_test_bucket_path, index=False)

    fig_path = OUT_DIR / f"{OUTPUT_PREFIX}_lambda_feature_sweep_top_val_bucket_trends.png"
    plot_top_bucket_trends(bucket_df_all, val_df.head(6), fig_path)

    print(f"[Done] wrote {val_path}")
    print(f"[Done] wrote {bucket_path}")
    print(f"[Done] wrote {calibration_path}")
    print(f"[Done] wrote {holdout_path}")
    print(f"[Done] wrote {holdout_bucket_path}")
    print(f"[Done] wrote {holdout_test_path}")
    print(f"[Done] wrote {holdout_test_bucket_path}")
    print(f"[Done] wrote {fold_path}")
    print(f"[Done] wrote {stability_path}")
    print(f"[Done] wrote {stable_test_path}")
    print(f"[Done] wrote {stable_test_bucket_path}")
    print(f"[Done] wrote {test_path}")
    print(f"[Done] wrote {test_bucket_path}")
    print(f"[Done] wrote {top_test_path}")
    print(f"[Done] wrote {top_test_bucket_path}")
    print(f"[Done] wrote {fig_path}")
    print("[TopVal]")
    print(
        val_df[
            [
                "mode",
                "window",
                "k",
                "spearman_mse",
                "spearman_mae",
                "bucket_mse_spearman",
                "bucket5_mse_lift_pct",
                "passes_spearman",
                "passes_bucket5",
                "passes_trend",
                "gate_candidate",
                "selection_score",
                "lambda_std",
                "lambda_iqr",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("[SelectedTest]")
    print(pd.DataFrame([test_row]).to_string(index=False))
    print("[CalibrationTopHoldout]")
    print(
        holdout_df[
            [
                "mode",
                "window",
                "k",
                "spearman_mse",
                "spearman_mae",
                "bucket_mse_spearman",
                "bucket5_mse_lift_pct",
                "gate_candidate",
                "selection_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("[HoldoutSelectedTest]")
    print(pd.DataFrame([holdout_test_row]).to_string(index=False))
    print("[FoldStabilityTop]")
    print(
        stability_df[
            [
                "mode",
                "window",
                "k",
                "fold_spearman_mean",
                "fold_spearman_min",
                "fold_bucket5_lift_mean",
                "fold_bucket5_lift_min",
                "positive_bucket5_fraction",
                "stable_candidate",
                "stability_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("[FoldStableSelectedTest]")
    print(pd.DataFrame([stable_test_row]).to_string(index=False))
    print("[TopValOnTest]")
    print(
        top_test_df[
            [
                "mode",
                "window",
                "k",
                "spearman_mse",
                "spearman_mae",
                "bucket_mse_spearman",
                "bucket5_mse_lift_pct",
                "bucket5_vs_bucket1_mse_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
