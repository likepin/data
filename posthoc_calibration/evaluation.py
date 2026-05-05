from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_ecl96_deltaA_manual_gate import bucket_indices, mse_mae

from .schedules import active_ratio_from_gamma, gamma_from_schedule
from .selection import resolve_mode_status


def pct_gain(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0
    return 100.0 * (before - after) / before


def mean_std_se(values: np.ndarray, ddof: int = 1) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if values.size <= 1:
        return mean, 0.0, 0.0
    std = float(values.std(ddof=ddof))
    se = float(std / math.sqrt(values.size))
    return mean, std, se


def score_validation_grid(
    schedules: list[dict],
    dynamic: np.ndarray,
    lambda_values: np.ndarray,
    static_dirs: list[Path],
    baseline_dirs: list[Path] | None,
    pred_file: str,
    true_file: str,
    profile_name: str,
    lambda_cfg: dict,
    active_eps: float,
    progress_stride: int = 10,
) -> pd.DataFrame:
    n_samples, pred_len, n_vars = dynamic.shape
    if baseline_dirs is not None and len(static_dirs) != len(baseline_dirs):
        raise RuntimeError(f"Projection dir mismatch: static={len(static_dirs)} baseline={len(baseline_dirs)}")

    n_proj = len(static_dirs)
    n_sched = len(schedules)
    gamma_matrix = np.stack([gamma_from_schedule(lambda_values, s) for s in schedules], axis=0).astype(np.float32)
    static_mse_proj = np.zeros((n_proj,), dtype=np.float64)
    static_mae_proj = np.zeros((n_proj,), dtype=np.float64)
    baseline_mse_proj = np.zeros((n_proj,), dtype=np.float64)
    baseline_mae_proj = np.zeros((n_proj,), dtype=np.float64)
    post_mse_proj = np.zeros((n_sched, n_proj), dtype=np.float64)
    post_mae_proj = np.zeros((n_sched, n_proj), dtype=np.float64)

    for projection, static_dir in enumerate(static_dirs):
        pred_static = np.load(static_dir / pred_file, mmap_mode="r")
        true_static = np.load(static_dir / true_file, mmap_mode="r")
        expected_shape = (n_samples, pred_len, n_vars)
        if pred_static.shape != expected_shape:
            raise RuntimeError(f"Unexpected static pred shape in {static_dir}: {pred_static.shape}, expected {expected_shape}")

        static_err = np.asarray(true_static, dtype=np.float32) - np.asarray(pred_static, dtype=np.float32)
        static_mse_proj[projection], static_mae_proj[projection] = mse_mae(static_err)
        if baseline_dirs is not None:
            baseline_dir = baseline_dirs[projection]
            pred_baseline = np.load(baseline_dir / pred_file, mmap_mode="r")
            true_baseline = np.load(baseline_dir / true_file, mmap_mode="r")
            if pred_baseline.shape != expected_shape:
                raise RuntimeError(f"Unexpected baseline pred shape in {baseline_dir}: {pred_baseline.shape}, expected {expected_shape}")
            baseline_err = np.asarray(true_baseline, dtype=np.float32) - np.asarray(pred_baseline, dtype=np.float32)
            baseline_mse_proj[projection], baseline_mae_proj[projection] = mse_mae(baseline_err)
            del baseline_err

        for idx, gamma in enumerate(gamma_matrix):
            post_err = static_err - gamma.reshape(-1, 1, 1) * dynamic
            post_mse_proj[idx, projection], post_mae_proj[idx, projection] = mse_mae(post_err)

        del static_err

    static_mse_mean, static_mse_std, static_mse_se = mean_std_se(static_mse_proj)
    static_mae_mean, static_mae_std, static_mae_se = mean_std_se(static_mae_proj)
    if baseline_dirs is not None:
        baseline_mse_mean, baseline_mse_std, baseline_mse_se = mean_std_se(baseline_mse_proj)
        baseline_mae_mean, baseline_mae_std, baseline_mae_se = mean_std_se(baseline_mae_proj)
    else:
        baseline_mse_mean = baseline_mse_std = baseline_mse_se = float("nan")
        baseline_mae_mean = baseline_mae_std = baseline_mae_se = float("nan")

    rows = []
    for idx, schedule in enumerate(schedules):
        if progress_stride > 0 and ((idx + 1) % progress_stride == 0 or idx + 1 == n_sched):
            print(f"[ValGrid] {idx + 1}/{n_sched}", flush=True)
        gamma = gamma_matrix[idx]
        post_mse_mean, post_mse_std, post_mse_se = mean_std_se(post_mse_proj[idx])
        post_mae_mean, post_mae_std, post_mae_se = mean_std_se(post_mae_proj[idx])
        rows.append(
            {
                "profile": profile_name,
                "split": "val",
                "lambda_mode": lambda_cfg["mode"],
                "lambda_window": lambda_cfg["window"],
                "lambda_k": lambda_cfg["k"],
                "lambda_transform": lambda_cfg.get("lambda_transform", "raw"),
                **schedule,
                "gamma_mean": float(gamma.mean()),
                "gamma_min_actual": float(gamma.min()),
                "gamma_max_actual": float(gamma.max()),
                "gamma_above_min_fraction": float(np.mean(gamma > float(schedule["gamma_min"]) + 1e-8)),
                "active_ratio": active_ratio_from_gamma(gamma, gamma_floor=float(schedule["gamma_min"]), active_eps=active_eps),
                "static_mse": static_mse_mean,
                "static_mse_std": static_mse_std,
                "static_mse_se": static_mse_se,
                "static_mae": static_mae_mean,
                "static_mae_std": static_mae_std,
                "static_mae_se": static_mae_se,
                "baseline_mse": baseline_mse_mean,
                "baseline_mse_std": baseline_mse_std,
                "baseline_mse_se": baseline_mse_se,
                "baseline_mae": baseline_mae_mean,
                "baseline_mae_std": baseline_mae_std,
                "baseline_mae_se": baseline_mae_se,
                "posthoc_mse": post_mse_mean,
                "posthoc_mse_std": post_mse_std,
                "posthoc_mse_se": post_mse_se,
                "posthoc_mae": post_mae_mean,
                "posthoc_mae_std": post_mae_std,
                "posthoc_mae_se": post_mae_se,
                "mse_gain_pct": pct_gain(static_mse_mean, post_mse_mean),
                "mae_gain_pct": pct_gain(static_mae_mean, post_mae_mean),
            }
        )
    return pd.DataFrame(rows).sort_values(["posthoc_mse", "posthoc_mae", "gamma_max", "q_low", "q_high"]).reset_index(drop=True)


def evaluate_selected_schedule(
    schedule: dict,
    dynamic: np.ndarray,
    lambda_values: np.ndarray,
    static_dirs: list[Path],
    pred_file: str,
    true_file: str,
    profile_name: str,
    split: str,
    lambda_cfg: dict,
    active_eps: float,
    active_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_samples, pred_len, n_vars = dynamic.shape
    gamma = gamma_from_schedule(lambda_values, schedule)
    active_ratio = active_ratio_from_gamma(gamma, gamma_floor=float(schedule["gamma_min"]), active_eps=active_eps)
    mode_status, mode_reason = resolve_mode_status(
        passed_selection=bool(schedule.get("passes_selection", True)),
        active_ratio=active_ratio,
        active_cutoff=active_cutoff,
        active_eps=active_eps,
    )
    buckets = bucket_indices(lambda_values, 5)
    all_rows = []
    bucket_rows = []

    for projection, directory in enumerate(static_dirs):
        pred = np.load(directory / pred_file, mmap_mode="r")
        true = np.load(directory / true_file, mmap_mode="r")
        if pred.shape != (n_samples, pred_len, n_vars):
            raise RuntimeError(f"Unexpected pred shape in {directory}: {pred.shape}, expected {(n_samples, pred_len, n_vars)}")
        static_err = np.asarray(true, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        post_err = static_err - gamma.reshape(-1, 1, 1) * dynamic
        static_mse, static_mae = mse_mae(static_err)
        post_mse, post_mae = mse_mae(post_err)
        base = {
            "profile": profile_name,
            "split": split,
            "projection": projection,
            "lambda_mode": lambda_cfg["mode"],
            "lambda_window": lambda_cfg["window"],
            "lambda_k": lambda_cfg["k"],
            "lambda_transform": lambda_cfg.get("lambda_transform", "raw"),
            **{k: schedule[k] for k in ("q_low", "q_high", "q_low_value", "q_high_value", "gamma_min", "gamma_max")},
            "gamma_mean": float(gamma.mean()),
            "gamma_min_actual": float(gamma.min()),
            "gamma_max_actual": float(gamma.max()),
            "gamma_above_min_fraction": float(np.mean(gamma > float(schedule["gamma_min"]) + 1e-8)),
            "active_ratio": active_ratio,
            "mode_status": mode_status,
            "mode_reason": mode_reason,
            "static_mse": static_mse,
            "posthoc_mse": post_mse,
            "mse_gain_pct": pct_gain(static_mse, post_mse),
            "static_mae": static_mae,
            "posthoc_mae": post_mae,
            "mae_gain_pct": pct_gain(static_mae, post_mae),
        }
        all_rows.append(base)

        for bucket_id, idx in enumerate(buckets, start=1):
            idx = np.asarray(idx, dtype=np.int64)
            s_mse, s_mae = mse_mae(static_err[idx])
            p_mse, p_mae = mse_mae(post_err[idx])
            bucket_rows.append(
                {
                    **base,
                    "bucket": bucket_id,
                    "n": int(idx.size),
                    "lambda_min": float(lambda_values[idx].min()),
                    "lambda_mean": float(lambda_values[idx].mean()),
                    "lambda_max": float(lambda_values[idx].max()),
                    "gamma_bucket_mean": float(gamma[idx].mean()),
                    "static_mse": s_mse,
                    "posthoc_mse": p_mse,
                    "mse_gain_pct": pct_gain(s_mse, p_mse),
                    "static_mae": s_mae,
                    "posthoc_mae": p_mae,
                    "mae_gain_pct": pct_gain(s_mae, p_mae),
                }
            )

    all_df = pd.DataFrame(all_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    group_keys = [
        "profile",
        "split",
        "mode_status",
        "mode_reason",
        "lambda_mode",
        "lambda_window",
        "lambda_k",
        "lambda_transform",
        "q_low",
        "q_high",
        "gamma_min",
        "gamma_max",
    ]
    all_summary = all_df.drop(columns=["projection"]).groupby(group_keys, as_index=False).mean(numeric_only=True)
    bucket_summary = bucket_df.drop(columns=["projection"]).groupby(group_keys + ["bucket"], as_index=False).mean(numeric_only=True)
    return all_summary, bucket_summary
