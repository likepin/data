from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    return float(np.sum(rx * ry) / denom) if denom > 0 else 0.0


def empirical_percentile_by_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        raise ValueError("Cannot rank-transform lambda with an empty reference split.")
    sorted_ref = np.sort(ref, kind="mergesort")
    ranks = np.searchsorted(sorted_ref, np.asarray(values, dtype=np.float64), side="right")
    return np.clip(ranks / float(sorted_ref.size), 0.0, 1.0).astype(np.float32)


def transform_lambda_splits(lambda_splits: dict[str, np.ndarray], transform: str) -> dict[str, np.ndarray]:
    if transform == "raw":
        return {key: np.asarray(value, dtype=np.float32) for key, value in lambda_splits.items()}
    if transform != "rank":
        raise ValueError(f"Unsupported lambda transform: {transform}")
    reference = np.asarray(lambda_splits["val"], dtype=np.float32)
    return {
        key: empirical_percentile_by_reference(np.asarray(value, dtype=np.float32), reference)
        for key, value in lambda_splits.items()
    }


def lambda_rank_diagnostics(
    raw_splits: dict[str, np.ndarray],
    transformed_splits: dict[str, np.ndarray],
    active_ratios: list[float],
) -> pd.DataFrame:
    rows = []
    for split, raw_values in raw_splits.items():
        raw = np.asarray(raw_values, dtype=np.float64)
        transformed = np.asarray(transformed_splits[split], dtype=np.float64)
        raw_max = float(np.max(raw))
        top_tie = np.isclose(raw, raw_max, rtol=1e-7, atol=1e-8)
        base = {
            "split": split,
            "n": int(raw.size),
            "raw_min": float(np.min(raw)),
            "raw_q50": float(np.quantile(raw, 0.50)),
            "raw_q80": float(np.quantile(raw, 0.80)),
            "raw_q90": float(np.quantile(raw, 0.90)),
            "raw_q95": float(np.quantile(raw, 0.95)),
            "raw_q99": float(np.quantile(raw, 0.99)),
            "raw_max": raw_max,
            "raw_top_tie_count": int(top_tie.sum()),
            "raw_top_tie_rate": float(top_tie.mean()),
            "rank_min": float(np.min(transformed)),
            "rank_q50": float(np.quantile(transformed, 0.50)),
            "rank_q80": float(np.quantile(transformed, 0.80)),
            "rank_q90": float(np.quantile(transformed, 0.90)),
            "rank_q95": float(np.quantile(transformed, 0.95)),
            "rank_q99": float(np.quantile(transformed, 0.99)),
            "rank_max": float(np.max(transformed)),
        }
        if active_ratios:
            for ratio in active_ratios:
                threshold = 1.0 - float(ratio)
                active = transformed > threshold
                rows.append(
                    {
                        **base,
                        "active_ratio_target": float(ratio),
                        "active_threshold": threshold,
                        "active_count": int(active.sum()),
                        "active_ratio_actual": float(active.mean()),
                        "active_top_tie_count": int(np.logical_and(active, top_tie).sum()),
                    }
                )
        else:
            rows.append(
                {
                    **base,
                    "active_ratio_target": float("nan"),
                    "active_threshold": float("nan"),
                    "active_count": 0,
                    "active_ratio_actual": 0.0,
                    "active_top_tie_count": 0,
                }
            )
    return pd.DataFrame(rows)


def load_timestamps(data_csv: Path, date_col: str | None, header_mode: str, sep: str) -> pd.Series | None:
    if date_col is None:
        return None
    header = 0 if header_mode == "infer" else None
    df = pd.read_csv(data_csv, header=header, sep=sep, usecols=[date_col])
    return df[date_col].astype(str)


def saturated_windows(
    *,
    split: str,
    raw_values: np.ndarray,
    transformed_values: np.ndarray,
    sample_start_rows: np.ndarray,
    seq_len: int,
    active_ratios: list[float],
    timestamps: pd.Series | None,
) -> pd.DataFrame:
    raw = np.asarray(raw_values, dtype=np.float64)
    transformed = np.asarray(transformed_values, dtype=np.float64)
    starts = np.asarray(sample_start_rows, dtype=np.int64)
    raw_max = float(np.max(raw))
    top_tie = np.isclose(raw, raw_max, rtol=1e-7, atol=1e-8)
    active_masks = {float(ratio): transformed > (1.0 - float(ratio)) for ratio in active_ratios}
    keep = top_tie.copy()
    for mask in active_masks.values():
        keep |= mask
    rows = []
    for sample_index in np.where(keep)[0]:
        start = int(starts[sample_index])
        context_end = start + int(seq_len) - 1
        target_start = start + int(seq_len)
        row = {
            "split": split,
            "sample_index": int(sample_index),
            "data_row_start": start,
            "data_row_context_end": context_end,
            "data_row_target_start": target_start,
            "context_end_timestamp": _timestamp_at(timestamps, context_end),
            "target_start_timestamp": _timestamp_at(timestamps, target_start),
            "lambda_raw": float(raw[sample_index]),
            "lambda_rank": float(transformed[sample_index]),
            "is_top_raw_tie": bool(top_tie[sample_index]),
        }
        for ratio, mask in active_masks.items():
            row[f"active_at_p_{ratio:g}"] = bool(mask[sample_index])
        rows.append(row)
    return pd.DataFrame(rows)


def _timestamp_at(timestamps: pd.Series | None, index: int) -> str | None:
    if timestamps is None or index < 0 or index >= len(timestamps):
        return None
    return str(timestamps.iloc[index])


def static_sample_errors(
    static_dirs: list[Path],
    pred_file: str,
    true_file: str,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if not static_dirs:
        raise ValueError("No static result dirs available for diagnostics.")
    first_pred = np.load(static_dirs[0] / pred_file, mmap_mode="r")
    n_samples = int(first_pred.shape[0])
    mse = np.zeros(n_samples, dtype=np.float64)
    mae = np.zeros(n_samples, dtype=np.float64)
    for directory in static_dirs:
        pred = np.load(directory / pred_file, mmap_mode="r")
        true = np.load(directory / true_file, mmap_mode="r")
        if pred.shape != true.shape:
            raise RuntimeError(f"Pred/true shape mismatch in {directory}: {pred.shape} vs {true.shape}")
        for start in range(0, n_samples, int(chunk_size)):
            end = min(start + int(chunk_size), n_samples)
            err = np.asarray(true[start:end], dtype=np.float32) - np.asarray(pred[start:end], dtype=np.float32)
            mse[start:end] += np.mean(err * err, axis=(1, 2))
            mae[start:end] += np.mean(np.abs(err), axis=(1, 2))
    scale = float(len(static_dirs))
    return mse / scale, mae / scale


def active_ratio_fold_consistency(
    lambda_values: np.ndarray,
    mse: np.ndarray,
    mae: np.ndarray,
    active_ratios: list[float],
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lam = np.asarray(lambda_values, dtype=np.float64)
    mse = np.asarray(mse, dtype=np.float64)
    mae = np.asarray(mae, dtype=np.float64)
    folds = np.array_split(np.arange(lam.size), int(n_folds))
    detail_rows = []
    for ratio in active_ratios:
        threshold = 1.0 - float(ratio)
        for fold_id, idx in enumerate(folds, start=1):
            idx = np.asarray(idx, dtype=np.int64)
            active = lam[idx] > threshold
            active_count = int(active.sum())
            fold_mse = mse[idx]
            fold_mae = mae[idx]
            row = {
                "active_ratio_target": float(ratio),
                "active_threshold": threshold,
                "fold_id": fold_id,
                "fold_n": int(idx.size),
                "active_count": active_count,
                "active_ratio_actual": float(active.mean()) if idx.size else 0.0,
                "fold_spearman_mse": spearman(lam[idx], fold_mse),
                "fold_spearman_mae": spearman(lam[idx], fold_mae),
                "global_mse": float(fold_mse.mean()),
                "global_mae": float(fold_mae.mean()),
            }
            if active_count > 0:
                active_mse = float(fold_mse[active].mean())
                active_mae = float(fold_mae[active].mean())
                row.update(
                    {
                        "active_mse": active_mse,
                        "active_mae": active_mae,
                        "active_mse_lift_pct": _pct_delta(row["global_mse"], active_mse),
                        "active_mae_lift_pct": _pct_delta(row["global_mae"], active_mae),
                    }
                )
            else:
                row.update(
                    {
                        "active_mse": float("nan"),
                        "active_mae": float("nan"),
                        "active_mse_lift_pct": float("nan"),
                        "active_mae_lift_pct": float("nan"),
                    }
                )
            detail_rows.append(row)
    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for ratio, sub in detail.groupby("active_ratio_target", sort=True):
        valid_lift = sub["active_mse_lift_pct"].dropna().to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "active_ratio_target": float(ratio),
                "folds": int(sub.shape[0]),
                "active_count_total": int(sub["active_count"].sum()),
                "active_ratio_actual_mean": float(sub["active_ratio_actual"].mean()),
                "fold_spearman_mse_mean": float(sub["fold_spearman_mse"].mean()),
                "fold_spearman_mse_min": float(sub["fold_spearman_mse"].min()),
                "fold_spearman_mae_mean": float(sub["fold_spearman_mae"].mean()),
                "fold_spearman_mae_min": float(sub["fold_spearman_mae"].min()),
                "active_mse_lift_mean": float(np.nanmean(valid_lift)) if valid_lift.size else float("nan"),
                "active_mse_lift_min": float(np.nanmin(valid_lift)) if valid_lift.size else float("nan"),
                "positive_lift_fraction": float(np.mean(valid_lift > 0.0)) if valid_lift.size else 0.0,
                "nonempty_fold_fraction": float(np.mean(sub["active_count"].to_numpy(dtype=np.int64) > 0)),
            }
        )
    return pd.DataFrame(summary_rows), detail


def lambda_quality_metrics(
    raw_values: np.ndarray,
    transformed_values: np.ndarray,
    static_mse: np.ndarray,
    static_mae: np.ndarray,
    active_ratios: list[float],
) -> dict:
    raw = np.asarray(raw_values, dtype=np.float64)
    transformed = np.asarray(transformed_values, dtype=np.float64)
    mse = np.asarray(static_mse, dtype=np.float64)
    mae = np.asarray(static_mae, dtype=np.float64)
    raw_max = float(np.max(raw))
    top_tie = np.isclose(raw, raw_max, rtol=1e-7, atol=1e-8)
    q80 = float(np.quantile(raw, 0.80))
    q90 = float(np.quantile(raw, 0.90))
    q95 = float(np.quantile(raw, 0.95))
    q99 = float(np.quantile(raw, 0.99))
    rows = {
        "lambda_raw_min": float(np.min(raw)),
        "lambda_raw_q50": float(np.quantile(raw, 0.50)),
        "lambda_raw_q80": q80,
        "lambda_raw_q90": q90,
        "lambda_raw_q95": q95,
        "lambda_raw_q99": q99,
        "lambda_raw_max": raw_max,
        "lambda_raw_iqr": float(np.quantile(raw, 0.75) - np.quantile(raw, 0.25)),
        "lambda_top_tie_rate": float(top_tie.mean()),
        "lambda_q99_q80_spread": float(q99 - q80),
        "residual_spearman_mse": spearman(transformed, mse),
        "residual_spearman_mae": spearman(transformed, mae),
    }
    best_ratio = None
    best_lift = -np.inf
    for ratio in active_ratios:
        ratio = float(ratio)
        active = transformed > (1.0 - ratio)
        active_count = int(active.sum())
        if active_count:
            mse_lift = _pct_delta(float(mse.mean()), float(mse[active].mean()))
            mae_lift = _pct_delta(float(mae.mean()), float(mae[active].mean()))
        else:
            mse_lift = float("nan")
            mae_lift = float("nan")
        rows[f"p_{ratio:g}_active_count"] = active_count
        rows[f"p_{ratio:g}_active_ratio_actual"] = float(active.mean())
        rows[f"p_{ratio:g}_mse_lift_pct"] = mse_lift
        rows[f"p_{ratio:g}_mae_lift_pct"] = mae_lift
        if np.isfinite(mse_lift) and mse_lift > best_lift:
            best_lift = float(mse_lift)
            best_ratio = ratio
    rows["best_active_ratio_by_mse_lift"] = float(best_ratio) if best_ratio is not None else float("nan")
    rows["best_active_mse_lift_pct"] = best_lift if np.isfinite(best_lift) else float("nan")
    return rows


def residual_complexity_alignment_frame(
    *,
    split: str,
    raw_values: np.ndarray,
    transformed_values: np.ndarray,
    static_mse: np.ndarray,
    static_mae: np.ndarray,
    active_ratios: list[float],
    sample_start_rows: np.ndarray,
    seq_len: int,
    timestamps: pd.Series | None,
    n_folds: int,
) -> pd.DataFrame:
    raw = np.asarray(raw_values, dtype=np.float64)
    transformed = np.asarray(transformed_values, dtype=np.float64)
    mse = np.asarray(static_mse, dtype=np.float64)
    mae = np.asarray(static_mae, dtype=np.float64)
    starts = np.asarray(sample_start_rows, dtype=np.int64)
    folds = np.array_split(np.arange(raw.size), int(n_folds))
    fold_id = np.zeros(raw.size, dtype=np.int64)
    for idx, fold in enumerate(folds, start=1):
        fold_id[np.asarray(fold, dtype=np.int64)] = idx
    rows = []
    for sample_index in range(raw.size):
        start = int(starts[sample_index])
        context_end = start + int(seq_len) - 1
        target_start = start + int(seq_len)
        row = {
            "split": split,
            "sample_index": sample_index,
            "fold_id": int(fold_id[sample_index]),
            "data_row_start": start,
            "data_row_context_end": context_end,
            "data_row_target_start": target_start,
            "context_end_timestamp": _timestamp_at(timestamps, context_end),
            "target_start_timestamp": _timestamp_at(timestamps, target_start),
            "lambda_raw": float(raw[sample_index]),
            "lambda_rank": float(transformed[sample_index]),
            "static_mse": float(mse[sample_index]),
            "static_mae": float(mae[sample_index]),
        }
        for ratio in active_ratios:
            row[f"active_at_p_{float(ratio):g}"] = bool(transformed[sample_index] > (1.0 - float(ratio)))
        rows.append(row)
    return pd.DataFrame(rows)


def write_residual_complexity_plot(
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str,
    x_col: str = "lambda_rank",
    y_col: str = "static_mse",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    scatter = ax.scatter(
        frame[x_col],
        frame[y_col],
        c=frame["fold_id"],
        cmap="viridis",
        s=10,
        alpha=0.55,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("fold_id")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _pct_delta(reference: float, value: float) -> float:
    if abs(reference) < 1e-12:
        return 0.0
    return 100.0 * (value - reference) / reference
