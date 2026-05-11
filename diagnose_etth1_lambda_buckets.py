from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posthoc_ecl96_deltaA_manual_gate import build_dynamic_cache, find_result_dirs, mse_mae


RESULT_ROOT = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\results")
INTERFACE_DIR = Path(r"C:\Users\cyl\Desktop\data\interfaces\ETTh1_graph_interface_cmiknn_ridgebase_sparse")
DATA_CSV = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\dataset\ETTh1.csv")
OUT_DIR = Path(r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\etth196_staticcausal_validation_grid")
BASELINE_PATTERN = "etth196_validate_baseline_itr3_*projection_*"
STATIC_PATTERN = "etth196_validate_static_anchor_itr3_*projection_*"
SELECTED_TEST_CSV = OUT_DIR / "etth196_staticcausal_late_ramp_test_selected.csv"


def make_args() -> SimpleNamespace:
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
        eval_split="test",
        pred_file="pred.npy",
        true_file="true.npy",
        output_prefix="unused",
        progress_every=1000,
    )


def describe_lambda(name: str, values: np.ndarray) -> dict:
    qs = {
        f"q{int(q * 100):02d}": float(np.quantile(values, q))
        for q in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    }
    return {
        "split": name,
        "n": int(values.shape[0]),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        **qs,
    }


def bucket_indices(lambda_values: np.ndarray, n_buckets: int = 5) -> list[np.ndarray]:
    order = np.argsort(lambda_values, kind="mergesort")
    return [np.asarray(idx, dtype=np.int64) for idx in np.array_split(order, n_buckets)]


def schedule_gamma(lambda_eval: np.ndarray, lambda_schedule: np.ndarray, selected: pd.Series) -> np.ndarray:
    q_low_value = float(np.quantile(lambda_schedule, float(selected["q_low"])))
    q_high_value = float(np.quantile(lambda_schedule, float(selected["q_high"])))
    weight = np.clip((lambda_eval - q_low_value) / (q_high_value - q_low_value), 0.0, 1.0)
    return (
        float(selected["gamma_min"]) + (float(selected["gamma_max"]) - float(selected["gamma_min"])) * weight
    ).astype(np.float32)


def load_projection_dirs(pattern: str, pred_file: str = "pred.npy", true_file: str = "true.npy") -> list[Path]:
    return find_result_dirs(RESULT_ROOT, pattern, pred_file=pred_file, true_file=true_file)


def bucket_metric_rows(
    *,
    lambda_eval: np.ndarray,
    baseline_dirs: list[Path],
    static_dirs: list[Path],
    dynamic: np.ndarray,
    gamma: np.ndarray,
) -> pd.DataFrame:
    buckets = bucket_indices(lambda_eval, n_buckets=5)
    rows = []
    gamma_values = gamma.reshape(-1, 1, 1)
    for bucket_id, idx in enumerate(buckets, start=1):
        projection_rows = []
        for projection, (base_dir, static_dir) in enumerate(zip(baseline_dirs, static_dirs)):
            base_pred = np.load(base_dir / "pred.npy", mmap_mode="r")
            base_true = np.load(base_dir / "true.npy", mmap_mode="r")
            static_pred = np.load(static_dir / "pred.npy", mmap_mode="r")
            static_true = np.load(static_dir / "true.npy", mmap_mode="r")
            baseline_err = np.asarray(base_true[idx], dtype=np.float32) - np.asarray(base_pred[idx], dtype=np.float32)
            static_err = np.asarray(static_true[idx], dtype=np.float32) - np.asarray(static_pred[idx], dtype=np.float32)
            posthoc_err = static_err - gamma_values[idx] * dynamic[idx]
            baseline_mse, baseline_mae = mse_mae(baseline_err)
            static_mse, static_mae = mse_mae(static_err)
            posthoc_mse, posthoc_mae = mse_mae(posthoc_err)
            projection_rows.append(
                {
                    "projection": projection,
                    "baseline_mse": baseline_mse,
                    "baseline_mae": baseline_mae,
                    "static_mse": static_mse,
                    "static_mae": static_mae,
                    "posthoc_mse": posthoc_mse,
                    "posthoc_mae": posthoc_mae,
                }
            )
        df = pd.DataFrame(projection_rows)
        row = {
            "bucket": bucket_id,
            "n": int(idx.shape[0]),
            "lambda_min": float(lambda_eval[idx].min()),
            "lambda_mean": float(lambda_eval[idx].mean()),
            "lambda_max": float(lambda_eval[idx].max()),
            "gamma_mean": float(gamma[idx].mean()),
        }
        for key in ["baseline_mse", "baseline_mae", "static_mse", "static_mae", "posthoc_mse", "posthoc_mae"]:
            row[key] = float(df[key].mean())
        row["static_vs_baseline_mse_gain_pct"] = 100.0 * (row["baseline_mse"] - row["static_mse"]) / row["baseline_mse"]
        row["static_vs_baseline_mae_gain_pct"] = 100.0 * (row["baseline_mae"] - row["static_mae"]) / row["baseline_mae"]
        row["posthoc_vs_baseline_mse_gain_pct"] = 100.0 * (row["baseline_mse"] - row["posthoc_mse"]) / row["baseline_mse"]
        row["posthoc_vs_baseline_mae_gain_pct"] = 100.0 * (row["baseline_mae"] - row["posthoc_mae"]) / row["baseline_mae"]
        row["posthoc_vs_static_mse_gain_pct"] = 100.0 * (row["static_mse"] - row["posthoc_mse"]) / row["static_mse"]
        row["posthoc_vs_static_mae_gain_pct"] = 100.0 * (row["static_mae"] - row["posthoc_mae"]) / row["static_mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_diagnostics(lambda_eval: np.ndarray, lambda_schedule: np.ndarray, bucket_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].hist(lambda_schedule, bins=40, alpha=0.65, label="val schedule", color="#4c78a8")
    axes[0, 0].hist(lambda_eval, bins=40, alpha=0.55, label="test", color="#f58518")
    axes[0, 0].set_title("ETTh1 Lambda Distribution")
    axes[0, 0].set_xlabel("lambda")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].legend()

    x = bucket_df["bucket"].to_numpy()
    axes[0, 1].plot(x, bucket_df["baseline_mse"], marker="o", label="baseline")
    axes[0, 1].plot(x, bucket_df["static_mse"], marker="o", label="static")
    axes[0, 1].plot(x, bucket_df["posthoc_mse"], marker="o", label="static+dynamic")
    axes[0, 1].set_title("Bucket MSE")
    axes[0, 1].set_xlabel("lambda bucket")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].legend()

    axes[1, 0].plot(x, bucket_df["baseline_mae"], marker="o", label="baseline")
    axes[1, 0].plot(x, bucket_df["static_mae"], marker="o", label="static")
    axes[1, 0].plot(x, bucket_df["posthoc_mae"], marker="o", label="static+dynamic")
    axes[1, 0].set_title("Bucket MAE")
    axes[1, 0].set_xlabel("lambda bucket")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].legend()

    axes[1, 1].bar(x - 0.15, bucket_df["static_vs_baseline_mse_gain_pct"], width=0.3, label="static vs baseline")
    axes[1, 1].bar(x + 0.15, bucket_df["posthoc_vs_baseline_mse_gain_pct"], width=0.3, label="posthoc vs baseline")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("MSE Gain vs Baseline (%)")
    axes[1, 1].set_xlabel("lambda bucket")
    axes[1, 1].set_ylabel("gain %")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = make_args()
    dynamic, lambda_eval, lambda_schedule, static_dirs = build_dynamic_cache(args)
    baseline_dirs = load_projection_dirs(BASELINE_PATTERN)
    if len(baseline_dirs) != len(static_dirs):
        raise RuntimeError(f"projection count mismatch: baseline={len(baseline_dirs)}, static={len(static_dirs)}")

    selected = pd.read_csv(SELECTED_TEST_CSV).iloc[0]
    gamma = schedule_gamma(lambda_eval=lambda_eval, lambda_schedule=lambda_schedule, selected=selected)

    lambda_stats = pd.DataFrame(
        [
            describe_lambda("val_schedule", lambda_schedule),
            describe_lambda("test", lambda_eval),
        ]
    )
    bucket_df = bucket_metric_rows(
        lambda_eval=lambda_eval,
        baseline_dirs=baseline_dirs,
        static_dirs=static_dirs,
        dynamic=dynamic,
        gamma=gamma,
    )

    lambda_path = OUT_DIR / "etth196_lambda_distribution_stats.csv"
    bucket_path = OUT_DIR / "etth196_lambda_bucket_diagnostics.csv"
    fig_path = OUT_DIR / "etth196_lambda_bucket_diagnostics.png"
    lambda_stats.to_csv(lambda_path, index=False)
    bucket_df.to_csv(bucket_path, index=False)
    plot_diagnostics(lambda_eval=lambda_eval, lambda_schedule=lambda_schedule, bucket_df=bucket_df, out_path=fig_path)

    print(f"[Done] wrote {lambda_path}")
    print(f"[Done] wrote {bucket_path}")
    print(f"[Done] wrote {fig_path}")
    print("[LambdaStats]")
    print(lambda_stats.to_string(index=False))
    print("[BucketDiagnostics]")
    print(
        bucket_df[
            [
                "bucket",
                "n",
                "lambda_min",
                "lambda_mean",
                "lambda_max",
                "baseline_mse",
                "static_mse",
                "posthoc_mse",
                "baseline_mae",
                "static_mae",
                "posthoc_mae",
                "static_vs_baseline_mse_gain_pct",
                "posthoc_vs_baseline_mse_gain_pct",
                "posthoc_vs_static_mse_gain_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
