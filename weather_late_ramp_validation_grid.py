from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from posthoc_ecl96_deltaA_manual_gate import build_dynamic_cache, mse_mae


RESULT_ROOT = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\results")
INTERFACE_DIR = Path(r"C:\Users\cyl\Desktop\data\interfaces\Weather_graph_interface_parcorr")
DATA_CSV = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\dataset\weather.csv")
OUT_DIR = Path(r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\weather96_staticcausal_validation_grid")
STATIC_PATTERN = "weather_96_96_staticcausal_softmax_itr3_*projection_*"


def make_args(eval_split: str, pred_file: str, true_file: str) -> SimpleNamespace:
    return SimpleNamespace(
        interface_dir=str(INTERFACE_DIR),
        result_root=str(RESULT_ROOT),
        data_csv=str(DATA_CSV),
        static_pattern=STATIC_PATTERN,
        n_buckets=5,
        pred_len=96,
        gammas=[0.03, 0.05, 0.08, 0.10],
        linear_gamma_min=0.03,
        linear_gamma_max=0.06,
        linear_q_low=0.80,
        linear_q_high=0.95,
        schedule_source="val",
        eval_split=eval_split,
        pred_file=pred_file,
        true_file=true_file,
        output_prefix="unused",
        progress_every=1000,
    )


def evaluate_schedule(
    *,
    dynamic: np.ndarray,
    lambda_eval: np.ndarray,
    lambda_schedule: np.ndarray,
    static_dirs: list[Path],
    pred_file: str,
    true_file: str,
    q_low: float,
    q_high: float,
    gamma_min: float,
    gamma_max: float,
) -> dict:
    q_low_value = float(np.quantile(lambda_schedule, q_low))
    q_high_value = float(np.quantile(lambda_schedule, q_high))
    if q_high_value <= q_low_value:
        raise ValueError(f"Invalid quantiles: q_low={q_low}, q_high={q_high}")
    linear_weight = np.clip((lambda_eval - q_low_value) / (q_high_value - q_low_value), 0.0, 1.0)
    gamma = (gamma_min + (gamma_max - gamma_min) * linear_weight).astype(np.float32)
    gamma_values = gamma.reshape(-1, 1, 1)

    projection_rows = []
    bucket5_mse_gain = []
    order = np.argsort(lambda_eval, kind="mergesort")
    bucket5 = np.asarray(np.array_split(order, 5)[4], dtype=np.int64)
    for projection, directory in enumerate(static_dirs):
        pred = np.load(directory / pred_file, mmap_mode="r")
        true = np.load(directory / true_file, mmap_mode="r")
        static_err = np.asarray(true, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        post_err = static_err - gamma_values * dynamic
        static_mse, static_mae = mse_mae(static_err)
        post_mse, post_mae = mse_mae(post_err)
        b_static_mse, _ = mse_mae(static_err[bucket5])
        b_post_mse, _ = mse_mae(post_err[bucket5])
        projection_rows.append(
            {
                "projection": projection,
                "static_mse": static_mse,
                "posthoc_mse": post_mse,
                "static_mae": static_mae,
                "posthoc_mae": post_mae,
                "bucket5_static_mse": b_static_mse,
                "bucket5_posthoc_mse": b_post_mse,
            }
        )
        bucket5_mse_gain.append(b_static_mse - b_post_mse)

    df = pd.DataFrame(projection_rows)
    static_mse = float(df["static_mse"].mean())
    post_mse = float(df["posthoc_mse"].mean())
    static_mae = float(df["static_mae"].mean())
    post_mae = float(df["posthoc_mae"].mean())
    bucket5_static_mse = float(df["bucket5_static_mse"].mean())
    bucket5_post_mse = float(df["bucket5_posthoc_mse"].mean())
    return {
        "q_low": q_low,
        "q_high": q_high,
        "q_low_value": q_low_value,
        "q_high_value": q_high_value,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "gamma_mean": float(gamma.mean()),
        "gamma_non_min_frac": float(np.mean(gamma > gamma_min + 1e-8)),
        "static_mse": static_mse,
        "posthoc_mse": post_mse,
        "mse_gain_abs": static_mse - post_mse,
        "mse_gain_pct": 100.0 * (static_mse - post_mse) / static_mse,
        "static_mae": static_mae,
        "posthoc_mae": post_mae,
        "mae_gain_abs": static_mae - post_mae,
        "mae_gain_pct": 100.0 * (static_mae - post_mae) / static_mae,
        "bucket5_static_mse": bucket5_static_mse,
        "bucket5_posthoc_mse": bucket5_post_mse,
        "bucket5_mse_gain_abs": bucket5_static_mse - bucket5_post_mse,
        "bucket5_mse_gain_pct": 100.0 * (bucket5_static_mse - bucket5_post_mse) / bucket5_static_mse,
    }


def evaluate_grid(eval_split: str, pred_file: str, true_file: str, schedules: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    args = make_args(eval_split=eval_split, pred_file=pred_file, true_file=true_file)
    dynamic, lambda_eval, lambda_schedule, static_dirs = build_dynamic_cache(args)
    rows = []
    for q_low, q_high, gamma_min, gamma_max in schedules:
        rows.append(
            {
                "eval_split": eval_split,
                **evaluate_schedule(
                    dynamic=dynamic,
                    lambda_eval=lambda_eval,
                    lambda_schedule=lambda_schedule,
                    static_dirs=static_dirs,
                    pred_file=pred_file,
                    true_file=true_file,
                    q_low=q_low,
                    q_high=q_high,
                    gamma_min=gamma_min,
                    gamma_max=gamma_max,
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_lows = [0.30, 0.50, 0.70, 0.80]
    q_highs = [0.70, 0.80, 0.90, 0.95]
    gamma_min = 0.03
    gamma_maxs = [0.04, 0.05, 0.06]
    schedules = [
        (q_low, q_high, gamma_min, gamma_max)
        for q_low in q_lows
        for q_high in q_highs
        if q_high > q_low
        for gamma_max in gamma_maxs
    ]

    val_df = evaluate_grid(
        eval_split="val",
        pred_file="val_pred.npy",
        true_file="val_true.npy",
        schedules=schedules,
    )
    val_path = OUT_DIR / "weather96_staticcausal_late_ramp_val_grid.csv"
    val_df.sort_values(["posthoc_mse", "posthoc_mae"]).to_csv(val_path, index=False)

    best = val_df.sort_values(["posthoc_mse", "posthoc_mae"]).iloc[0]
    best_schedule = [
        (
            float(best["q_low"]),
            float(best["q_high"]),
            float(best["gamma_min"]),
            float(best["gamma_max"]),
        )
    ]
    test_df = evaluate_grid(
        eval_split="test",
        pred_file="pred.npy",
        true_file="true.npy",
        schedules=best_schedule,
    )
    test_path = OUT_DIR / "weather96_staticcausal_late_ramp_test_selected.csv"
    test_df.to_csv(test_path, index=False)

    top_path = OUT_DIR / "weather96_staticcausal_late_ramp_val_top10.csv"
    val_df.sort_values(["posthoc_mse", "posthoc_mae"]).head(10).to_csv(top_path, index=False)

    print(f"[Done] wrote {val_path}")
    print(f"[Done] wrote {top_path}")
    print(f"[Done] wrote {test_path}")
    print("[BestVal]")
    print(best[["q_low", "q_high", "gamma_min", "gamma_max", "posthoc_mse", "mse_gain_pct", "posthoc_mae", "mae_gain_pct", "gamma_mean", "gamma_non_min_frac"]].to_string())
    print("[SelectedTest]")
    print(test_df.iloc[0][["q_low", "q_high", "gamma_min", "gamma_max", "posthoc_mse", "mse_gain_pct", "posthoc_mae", "mae_gain_pct", "gamma_mean", "gamma_non_min_frac"]].to_string())


if __name__ == "__main__":
    main()
