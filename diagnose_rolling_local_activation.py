from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def local_percentile(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    out = np.zeros(n, dtype=np.float64)
    half = int(window) // 2
    for i in range(n):
        if mode == "centered":
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
        elif mode == "trailing":
            lo = max(0, i - int(window) + 1)
            hi = i + 1
        else:
            raise ValueError(f"Unsupported rolling mode: {mode}")
        ref = np.sort(values[lo:hi], kind="mergesort")
        out[i] = np.searchsorted(ref, values[i], side="right") / float(ref.size)
    return out


def active_fold_distribution(active: np.ndarray, folds: np.ndarray) -> dict[str, float]:
    fold_ids = np.sort(np.unique(folds))
    counts = np.array([np.sum(active & (folds == fold_id)) for fold_id in fold_ids], dtype=np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return {
            "active_fold_coverage": 0.0,
            "active_fold_entropy_norm": 0.0,
            "active_effective_folds": 0.0,
            "active_fold_concentration": 0.0,
        }
    p = counts / total
    nz = p[p > 0.0]
    entropy = float(-np.sum(nz * np.log(nz)))
    return {
        "active_fold_coverage": float(np.mean(counts > 0.0)),
        "active_fold_entropy_norm": float(entropy / np.log(len(fold_ids))) if len(fold_ids) > 1 else 0.0,
        "active_effective_folds": float(np.exp(entropy)),
        "active_fold_concentration": float(np.max(p)),
    }


def pct_delta(base: float, value: float) -> float:
    return 100.0 * (value - base) / base if base != 0.0 else float("nan")


def add_rows(
    *,
    frame: pd.DataFrame,
    percentiles: np.ndarray,
    rank_scope: str,
    window_n: int,
    active_ratios: list[float],
    summary_rows: list[dict],
    detail_rows: list[dict],
) -> None:
    folds = frame["fold_id"].to_numpy(dtype=np.int64)
    mse = frame["static_mse"].to_numpy(dtype=np.float64)
    mae = frame["static_mae"].to_numpy(dtype=np.float64)
    global_mse = float(mse.mean())
    global_mae = float(mae.mean())
    for ratio in active_ratios:
        active = percentiles > (1.0 - float(ratio))
        active_count = int(active.sum())
        active_mse = float(mse[active].mean()) if active_count else float("nan")
        active_mae = float(mae[active].mean()) if active_count else float("nan")
        summary_rows.append(
            {
                "rank_scope": rank_scope,
                "window_n": int(window_n),
                "active_ratio_target": float(ratio),
                "active_count": active_count,
                "active_ratio_actual": float(active.mean()),
                **active_fold_distribution(active, folds),
                "active_mse_mean": active_mse,
                "inactive_mse_mean": float(mse[~active].mean()) if (~active).any() else float("nan"),
                "active_vs_global_mse_lift_pct": pct_delta(global_mse, active_mse) if active_count else float("nan"),
                "active_mae_mean": active_mae,
                "inactive_mae_mean": float(mae[~active].mean()) if (~active).any() else float("nan"),
                "active_vs_global_mae_lift_pct": pct_delta(global_mae, active_mae) if active_count else float("nan"),
            }
        )
        for fold_id in np.sort(np.unique(folds)):
            fold_mask = folds == fold_id
            fold_active = active & fold_mask
            fold_mse = float(mse[fold_mask].mean())
            fold_mae = float(mae[fold_mask].mean())
            fold_active_count = int(fold_active.sum())
            fold_active_mse = float(mse[fold_active].mean()) if fold_active_count else float("nan")
            fold_active_mae = float(mae[fold_active].mean()) if fold_active_count else float("nan")
            detail_rows.append(
                {
                    "rank_scope": rank_scope,
                    "window_n": int(window_n),
                    "active_ratio_target": float(ratio),
                    "fold_id": int(fold_id),
                    "fold_n": int(fold_mask.sum()),
                    "active_count": fold_active_count,
                    "active_ratio_actual": float(fold_active_count / max(1, int(fold_mask.sum()))),
                    "fold_mse_mean": fold_mse,
                    "active_mse_mean": fold_active_mse,
                    "active_vs_fold_mse_lift_pct": pct_delta(fold_mse, fold_active_mse)
                    if fold_active_count
                    else float("nan"),
                    "fold_mae_mean": fold_mae,
                    "active_mae_mean": fold_active_mae,
                    "active_vs_fold_mae_lift_pct": pct_delta(fold_mae, fold_active_mae)
                    if fold_active_count
                    else float("nan"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose global-vs-rolling active-window localization.")
    parser.add_argument("--alignment-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--windows", default="168,336,504")
    parser.add_argument("--active-ratios", default="0.05,0.10,0.20")
    parser.add_argument("--rolling-mode", choices=["centered", "trailing"], default="centered")
    parser.add_argument("--raw-col", default="lambda_raw")
    parser.add_argument("--global-rank-col", default="lambda_rank")
    args = parser.parse_args()

    frame = pd.read_csv(args.alignment_csv)
    out_dir = args.out_dir or args.alignment_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.alignment_csv.stem.replace("_residual_complexity_alignment", "")
    active_ratios = parse_float_list(args.active_ratios)
    windows = parse_int_list(args.windows)

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    add_rows(
        frame=frame,
        percentiles=frame[args.global_rank_col].to_numpy(dtype=np.float64),
        rank_scope="global_val",
        window_n=int(len(frame)),
        active_ratios=active_ratios,
        summary_rows=summary_rows,
        detail_rows=detail_rows,
    )
    raw_values = frame[args.raw_col].to_numpy(dtype=np.float64)
    for window in windows:
        percentiles = local_percentile(raw_values, window=window, mode=args.rolling_mode)
        add_rows(
            frame=frame,
            percentiles=percentiles,
            rank_scope=f"{args.rolling_mode}_rolling_raw",
            window_n=int(window),
            active_ratios=active_ratios,
            summary_rows=summary_rows,
            detail_rows=detail_rows,
        )

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    summary_path = out_dir / f"{prefix}_rolling_local_activation_summary.csv"
    detail_path = out_dir / f"{prefix}_rolling_local_activation_by_fold.csv"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    print(f"[Done] wrote {summary_path}")
    print(f"[Done] wrote {detail_path}")


if __name__ == "__main__":
    main()
