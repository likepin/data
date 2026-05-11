from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score

from lambda_adequacy_audit import (
    DATA_ROOT,
    default_alpha_csv,
    load_input_shift_features,
    read_one_csv,
    run_prefix,
)
from posthoc_calibration.profiles import PROFILES, compute_selected_lambda_splits, dynamic_args
from posthoc_calibration.schedules import gamma_from_schedule
from posthoc_ecl96_deltaA_manual_gate import build_dynamic_cache


WINDOW_FEATURES = [
    "lambda_rank",
    "gamma_selected",
    "gamma_active",
    "dynamic_energy",
    "dynamic_abs_mean",
    "input_abs_mean",
    "input_sq_mean",
    "input_delta_abs_mean",
]

TARGET_FEATURES = [
    "lambda_rank",
    "gamma_selected",
    "gamma_active",
    "dynamic_energy_target",
    "dynamic_abs_mean_target",
    "input_abs_mean",
    "input_sq_mean",
    "input_delta_abs_mean",
    "alpha_shrunk",
    "reliability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit lightweight logistic probes for oracle-positive dynamic correction windows."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="weather96_static_pat3")
    parser.add_argument("--audit-tag", default="lambda_adequacy")
    parser.add_argument("--closed-loop-tag", default="full_guard_v2")
    parser.add_argument("--tag", default="lambda_gate_logistic_probe")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--eta-max", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--top-fracs", default="0.01,0.05,0.10,0.20")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--audit-dir", default="")
    parser.add_argument("--adaptive-alpha-csv", default="")
    parser.add_argument("--write-target-scores", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def finite_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for feature in features:
        if feature not in out:
            out[feature] = 0.0
    out[features] = out[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def standardize(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    train_x = train[features].astype(float)
    test_x = test[features].astype(float)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return ((train_x - mean) / std).to_numpy(), ((test_x - mean) / std).to_numpy(), mean, std


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan")


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan")


def safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    frame = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return float("nan")
    return float(frame["a"].corr(frame["b"], method=method))


def fit_probe(frame: pd.DataFrame, features: list[str], label_col: str, score_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = finite_features(frame, features)
    train = frame[frame["split"] == "val"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or test.empty:
        raise RuntimeError("Expected both val and test rows")

    y_train = (train[label_col].to_numpy(dtype=float) > 0.0).astype(int)
    y_test = (test[label_col].to_numpy(dtype=float) > 0.0).astype(int)
    x_train, x_test, mean, std = standardize(train, test, features)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=0,
    )
    model.fit(x_train, y_train)
    train_score = model.predict_proba(x_train)[:, 1]
    test_score = model.predict_proba(x_test)[:, 1]
    train[f"{score_prefix}_score"] = train_score
    test[f"{score_prefix}_score"] = test_score

    coef = pd.DataFrame(
        {
            "feature": features,
            "coef_standardized": model.coef_[0],
            "abs_coef": np.abs(model.coef_[0]),
            "train_mean": [float(mean[f]) for f in features],
            "train_std": [float(std[f]) for f in features],
        }
    ).sort_values("abs_coef", ascending=False)

    metrics = pd.DataFrame(
        [
            {
                "split": "val",
                "n": int(len(train)),
                "positive_rate": float(y_train.mean()),
                "roc_auc": safe_auc(y_train, train_score),
                "average_precision": safe_ap(y_train, train_score),
                "oracle_gain_mean": float(train[label_col].mean()),
            },
            {
                "split": "test",
                "n": int(len(test)),
                "positive_rate": float(y_test.mean()),
                "roc_auc": safe_auc(y_test, test_score),
                "average_precision": safe_ap(y_test, test_score),
                "oracle_gain_mean": float(test[label_col].mean()),
            },
        ]
    )
    scored = pd.concat([train, test], ignore_index=True)
    return coef, metrics, scored


def fit_gain_regressors(
    frame: pd.DataFrame,
    features: list[str],
    label_col: str,
    score_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = finite_features(frame, features)
    train = frame[frame["split"] == "val"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or test.empty:
        raise RuntimeError("Expected both val and test rows")

    y_train = train[label_col].to_numpy(dtype=float)
    y_test = test[label_col].to_numpy(dtype=float)
    x_train, x_test, mean, std = standardize(train, test, features)
    models = {
        "ridge": Ridge(alpha=1.0),
        "huber": HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=500),
    }

    coef_rows = []
    metric_rows = []
    scored_frames = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_scored = train.copy()
        test_scored = test.copy()
        score_col = f"{score_prefix}_{model_name}_pred_gain"
        train_scored[score_col] = train_pred
        test_scored[score_col] = test_pred
        train_scored["gain_model"] = model_name
        test_scored["gain_model"] = model_name
        scored_frames.extend([train_scored, test_scored])

        for split, y_true, y_pred in (("val", y_train, train_pred), ("test", y_test, test_pred)):
            metric_rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "n": int(len(y_true)),
                    "actual_gain_mean": float(np.mean(y_true)),
                    "pred_gain_mean": float(np.mean(y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "r2": float(r2_score(y_true, y_pred)),
                    "pearson": safe_corr(y_true, y_pred, "pearson"),
                    "spearman": safe_corr(y_true, y_pred, "spearman"),
                    "positive_rate": float(np.mean(y_true > 0.0)),
                }
            )

        coefficients = np.asarray(getattr(model, "coef_", np.zeros(len(features))), dtype=float)
        for feature, coef in zip(features, coefficients):
            coef_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "coef_standardized": float(coef),
                    "abs_coef": float(abs(coef)),
                    "train_mean": float(mean[feature]),
                    "train_std": float(std[feature]),
                }
            )

    coef = pd.DataFrame(coef_rows).sort_values(["model", "abs_coef"], ascending=[True, False])
    metrics = pd.DataFrame(metric_rows)
    scored = pd.concat(scored_frames, ignore_index=True)
    return coef, metrics, scored


def topk_capture(scored: pd.DataFrame, score_col: str, label_col: str, top_fracs: list[float]) -> pd.DataFrame:
    rows = []
    for split, group in scored.groupby("split", sort=False):
        labels = (group[label_col].to_numpy(dtype=float) > 0.0).astype(int)
        gains = group[label_col].to_numpy(dtype=float)
        scores = group[score_col].to_numpy(dtype=float)
        order = np.argsort(scores, kind="mergesort")[::-1]
        total_pos_gain = float(np.clip(gains, 0.0, None).sum())
        for frac in top_fracs:
            n_top = max(1, int(round(len(group) * frac)))
            idx = order[:n_top]
            rows.append(
                {
                    "split": split,
                    "top_frac": float(frac),
                    "top_n": int(n_top),
                    "positive_rate_top": float(labels[idx].mean()),
                    "positive_rate_all": float(labels.mean()),
                    "oracle_gain_mean_top": float(gains[idx].mean()),
                    "oracle_gain_mean_all": float(gains.mean()),
                    "positive_gain_capture_share": (
                        float(np.clip(gains[idx], 0.0, None).sum() / total_pos_gain)
                        if total_pos_gain > 0
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def gain_topk_capture(
    scored: pd.DataFrame,
    score_col: str,
    label_col: str,
    top_fracs: list[float],
    tail_frac: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for (model, split), group in scored.groupby(["gain_model", "split"], sort=False):
        gains = group[label_col].to_numpy(dtype=float)
        scores = group[score_col].to_numpy(dtype=float)
        order = np.argsort(scores, kind="mergesort")[::-1]
        all_tail_n = max(1, int(round(len(gains) * tail_frac)))
        all_tail = np.sort(gains)[:all_tail_n]
        total_pos_gain = float(np.clip(gains, 0.0, None).sum())
        for frac in top_fracs:
            n_top = max(1, int(round(len(group) * frac)))
            idx = order[:n_top]
            selected = gains[idx]
            selected_tail_n = max(1, int(round(len(selected) * tail_frac)))
            selected_tail = np.sort(selected)[:selected_tail_n]
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "top_frac": float(frac),
                    "top_n": int(n_top),
                    "pred_gain_mean_top": float(scores[idx].mean()),
                    "oracle_gain_mean_top": float(selected.mean()),
                    "oracle_gain_mean_all": float(gains.mean()),
                    "positive_rate_top": float(np.mean(selected > 0.0)),
                    "positive_rate_all": float(np.mean(gains > 0.0)),
                    "worst_5pct_gain_mean_top": float(selected_tail.mean()),
                    "worst_5pct_gain_mean_all": float(all_tail.mean()),
                    "p05_gain_top": float(np.quantile(selected, tail_frac)),
                    "p05_gain_all": float(np.quantile(gains, tail_frac)),
                    "positive_gain_capture_share": (
                        float(np.clip(selected, 0.0, None).sum() / total_pos_gain)
                        if total_pos_gain > 0
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def gain_quantile_bins(
    scored: pd.DataFrame,
    score_col: str,
    label_col: str,
    energy_col: str,
    n_bins: int = 10,
    tail_frac: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for (model, split), group in scored.groupby(["gain_model", "split"], sort=False):
        group = group.copy()
        group["_score"] = group[score_col].to_numpy(dtype=float)
        group["_gain"] = group[label_col].to_numpy(dtype=float)
        group = group.sort_values("_score", ascending=False).reset_index(drop=True)
        n = len(group)
        for bin_id, idx in enumerate(np.array_split(np.arange(n), n_bins), start=1):
            if len(idx) == 0:
                continue
            part = group.iloc[idx]
            gains = part["_gain"].to_numpy(dtype=float)
            tail_n = max(1, int(round(len(gains) * tail_frac)))
            tail = np.sort(gains)[:tail_n]
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "rank_bin": int(bin_id),
                    "rank_bin_label": f"top_{(bin_id - 1) * 100 // n_bins}_{bin_id * 100 // n_bins}pct",
                    "n": int(len(part)),
                    "pred_gain_mean": float(part["_score"].mean()),
                    "pred_gain_min": float(part["_score"].min()),
                    "pred_gain_max": float(part["_score"].max()),
                    "oracle_gain_mean": float(gains.mean()),
                    "positive_rate": float(np.mean(gains > 0.0)),
                    "nonzero_dynamic_rate": float(np.mean(part[energy_col].to_numpy(dtype=float) > 1e-12)),
                    "worst_5pct_gain_mean": float(tail.mean()),
                    "p05_gain": float(np.quantile(gains, tail_frac)),
                    "p50_gain": float(np.quantile(gains, 0.50)),
                    "p95_gain": float(np.quantile(gains, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def load_alpha(profile_name: str, alpha_csv_arg: str, n_vars: int) -> pd.DataFrame:
    alpha_csv = Path(alpha_csv_arg) if alpha_csv_arg else default_alpha_csv(profile_name)
    if alpha_csv is None or not alpha_csv.exists():
        return pd.DataFrame(
            {
                "target_index": np.arange(n_vars, dtype=np.int64),
                "alpha_shrunk": np.zeros(n_vars, dtype=np.float64),
                "reliability": np.zeros(n_vars, dtype=np.float64),
            }
        )
    alpha = pd.read_csv(alpha_csv)
    keep = [col for col in ["target_index", "alpha_shrunk", "reliability"] if col in alpha]
    alpha = alpha[keep].copy()
    if "alpha_shrunk" not in alpha:
        alpha["alpha_shrunk"] = 0.0
    if "reliability" not in alpha:
        alpha["reliability"] = 0.0
    return alpha


def targetwise_rows(
    profile_name: str,
    profile: dict,
    split: str,
    lambda_values: np.ndarray,
    schedule: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    print(f"[Stage] target-wise dynamic cache split={split}", flush=True)
    dynamic, _legacy_lambda, _schedule_lambda, static_dirs = build_dynamic_cache(
        dynamic_args(profile, split=split, pred_len=args.pred_len, progress_every=args.progress_every)
    )
    n_samples, pred_len, n_vars = dynamic.shape
    if len(lambda_values) != n_samples:
        raise RuntimeError(f"Lambda length mismatch for {split}: {len(lambda_values)} vs {n_samples}")

    dyn = np.asarray(dynamic, dtype=np.float32)
    dyn_sq_sum = np.square(dyn, dtype=np.float32).sum(axis=1, dtype=np.float64)
    dyn_abs_mean = np.abs(dyn).sum(axis=1, dtype=np.float64) / float(pred_len)
    gamma = gamma_from_schedule(lambda_values, schedule).astype(np.float64)
    gamma_floor = float(schedule["gamma_min"])

    align_sum = np.zeros((n_samples, n_vars), dtype=np.float64)
    pred_file = "val_pred.npy" if split == "val" else "pred.npy"
    true_file = "val_true.npy" if split == "val" else "true.npy"
    expected_shape = (n_samples, pred_len, n_vars)
    for projection, directory in enumerate(static_dirs):
        pred = np.load(Path(directory) / pred_file, mmap_mode="r")
        true = np.load(Path(directory) / true_file, mmap_mode="r")
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape in {directory}: {pred.shape}, expected {expected_shape}")
        err = np.asarray(true, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        align_sum += (err * dyn).sum(axis=1, dtype=np.float64)
        del err
        print(f"[Split:{split}] target-wise projection {projection + 1}/{len(static_dirs)}", flush=True)

    n_proj = len(static_dirs)
    count = float(n_proj * pred_len)
    unit_gain = (2.0 * align_sum - float(n_proj) * dyn_sq_sum) / count
    selected_gain = (
        2.0 * gamma[:, None] * align_sum - np.square(gamma)[:, None] * float(n_proj) * dyn_sq_sum
    ) / count
    eta_raw = align_sum / np.maximum(float(n_proj) * dyn_sq_sum, 1e-12)
    eta = np.clip(eta_raw, 0.0, float(args.eta_max))
    eta2_gain = (2.0 * eta * align_sum - np.square(eta) * float(n_proj) * dyn_sq_sum) / count

    input_features = load_input_shift_features(profile, split=split, n_samples=n_samples, seq_len=args.seq_len)
    alpha = load_alpha(profile_name, args.adaptive_alpha_csv, n_vars)
    alpha = alpha.set_index("target_index").reindex(np.arange(n_vars)).fillna(0.0).reset_index()

    lambda_rank = pd.Series(lambda_values).rank(pct=True, method="average").to_numpy(dtype=np.float64)
    rows = {
        "profile": np.repeat(profile_name, n_samples * n_vars),
        "split": np.repeat(split, n_samples * n_vars),
        "sample_id": np.repeat(np.arange(n_samples, dtype=np.int64), n_vars),
        "target_index": np.tile(np.arange(n_vars, dtype=np.int64), n_samples),
        "lambda_value": np.repeat(lambda_values.astype(np.float64), n_vars),
        "lambda_rank": np.repeat(lambda_rank, n_vars),
        "gamma_selected": np.repeat(gamma, n_vars),
        "gamma_active": np.repeat(gamma > (gamma_floor + 1e-6), n_vars),
        "dynamic_energy_target": (dyn_sq_sum / float(pred_len)).reshape(-1),
        "dynamic_abs_mean_target": dyn_abs_mean.reshape(-1),
        "input_abs_mean": np.repeat(input_features["input_abs_mean"].to_numpy(dtype=np.float64), n_vars),
        "input_sq_mean": np.repeat(input_features["input_sq_mean"].to_numpy(dtype=np.float64), n_vars),
        "input_delta_abs_mean": np.repeat(input_features["input_delta_abs_mean"].to_numpy(dtype=np.float64), n_vars),
        "alpha_shrunk": np.tile(alpha["alpha_shrunk"].to_numpy(dtype=np.float64), n_samples),
        "reliability": np.tile(alpha["reliability"].to_numpy(dtype=np.float64), n_samples),
        "oracle_unit_mse_gain": unit_gain.reshape(-1),
        "oracle_eta2_mse_gain": eta2_gain.reshape(-1),
        "selected_gamma_mse_gain": selected_gain.reshape(-1),
    }
    return pd.DataFrame(rows)


def target_summary(scored: pd.DataFrame, score_col: str) -> pd.DataFrame:
    grouped = scored.groupby(["split", "target_index"], sort=False)
    return grouped.agg(
        n=("oracle_unit_mse_gain", "size"),
        positive_rate=("oracle_unit_mse_gain", lambda x: float((x > 0).mean())),
        oracle_gain_mean=("oracle_unit_mse_gain", "mean"),
        model_score_mean=(score_col, "mean"),
        alpha_shrunk=("alpha_shrunk", "first"),
        reliability=("reliability", "first"),
        dynamic_energy_target=("dynamic_energy_target", "mean"),
    ).reset_index()


def write_gain_figures(
    out_dir: Path,
    prefix: str,
    window_gain_topk: pd.DataFrame,
    target_gain_topk: pd.DataFrame,
    window_gain_scored: pd.DataFrame,
    target_gain_scored: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[Warn] skip gain figures: {exc}", flush=True)
        return

    def frontier(frame: pd.DataFrame, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for (model, split), group in frame.groupby(["model", "split"], sort=False):
            ordered = group.sort_values("top_frac")
            label = f"{model}-{split}"
            ax.plot(
                ordered["worst_5pct_gain_mean_top"],
                ordered["oracle_gain_mean_top"],
                marker="o",
                label=label,
            )
            for _, row in ordered.iterrows():
                ax.annotate(
                    f"{float(row['top_frac']):.0%}",
                    (row["worst_5pct_gain_mean_top"], row["oracle_gain_mean_top"]),
                    fontsize=8,
                )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("worst 5% mean gain in selected set")
        ax.set_ylabel("mean oracle gain in selected set")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    def top5_hist(frame: pd.DataFrame, score_prefix: str, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        test = frame[frame["split"] == "test"].copy()
        for model, group in test.groupby("gain_model", sort=False):
            score_col = f"{score_prefix}_{model}_pred_gain"
            order = group[score_col].to_numpy(dtype=float).argsort(kind="mergesort")[::-1]
            n_top = max(1, int(round(len(group) * 0.05)))
            selected = group.iloc[order[:n_top]]["oracle_unit_mse_gain"].to_numpy(dtype=float)
            ax.hist(selected, bins=60, alpha=0.45, density=True, label=f"{model} top 5%")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("oracle unit MSE gain")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    frontier(
        window_gain_topk,
        f"{prefix}: window gain risk-return frontier",
        out_dir / f"{prefix}_window_gain_risk_return_frontier.png",
    )
    frontier(
        target_gain_topk,
        f"{prefix}: target-wise gain risk-return frontier",
        out_dir / f"{prefix}_target_gain_risk_return_frontier.png",
    )
    top5_hist(
        window_gain_scored,
        "window_gain",
        f"{prefix}: window test top-5% gain distribution",
        out_dir / f"{prefix}_window_test_top5_gain_distribution.png",
    )
    top5_hist(
        target_gain_scored,
        "target_gain",
        f"{prefix}: target-wise test top-5% gain distribution",
        out_dir / f"{prefix}_target_test_top5_gain_distribution.png",
    )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(
    out_dir: Path,
    prefix: str,
    window_coef: pd.DataFrame,
    window_metrics: pd.DataFrame,
    target_coef: pd.DataFrame,
    target_metrics: pd.DataFrame,
    target_topk: pd.DataFrame,
    window_gain_coef: pd.DataFrame,
    window_gain_metrics: pd.DataFrame,
    window_gain_topk: pd.DataFrame,
    target_gain_coef: pd.DataFrame,
    target_gain_metrics: pd.DataFrame,
    target_gain_topk: pd.DataFrame,
    window_gain_bins: pd.DataFrame,
    target_gain_bins: pd.DataFrame,
) -> None:
    lines = [
        f"# {prefix} Logistic Gate Probe",
        "",
        "This is a diagnostic probe, not a final selection protocol. It trains on validation rows and reports test diagnostics only.",
        "",
        "## Window-Level Metrics",
        "",
        markdown_table(window_metrics),
        "",
        "## Window-Level Coefficients",
        "",
        markdown_table(window_coef),
        "",
        "## Target-Wise Metrics",
        "",
        markdown_table(target_metrics),
        "",
        "## Target-Wise Coefficients",
        "",
        markdown_table(target_coef),
        "",
        "## Target-Wise Top-K Capture",
        "",
        markdown_table(target_topk),
        "",
        "## Window-Level Gain Regression Metrics",
        "",
        markdown_table(window_gain_metrics),
        "",
        "## Window-Level Gain Regression Coefficients",
        "",
        markdown_table(window_gain_coef),
        "",
        "## Window-Level Gain Regression Top-K / CVaR",
        "",
        markdown_table(window_gain_topk),
        "",
        "## Target-Wise Gain Regression Metrics",
        "",
        markdown_table(target_gain_metrics),
        "",
        "## Target-Wise Gain Regression Coefficients",
        "",
        markdown_table(target_gain_coef),
        "",
        "## Target-Wise Gain Regression Top-K / CVaR",
        "",
        markdown_table(target_gain_topk),
        "",
        "## Window-Level Gain Quantile Bins",
        "",
        markdown_table(window_gain_bins),
        "",
        "## Target-Wise Gain Quantile Bins",
        "",
        markdown_table(target_gain_bins),
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    prefix = run_prefix(args.profile, args.tag)
    audit_prefix = run_prefix(args.profile, args.audit_tag)
    audit_dir = Path(args.audit_dir) if args.audit_dir else DATA_ROOT / "deltaA_signal_audit" / audit_prefix
    out_dir = Path(args.out_dir) if args.out_dir else DATA_ROOT / "deltaA_signal_audit" / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = audit_dir / f"{audit_prefix}_sample_scores.csv"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    sample = pd.read_csv(sample_path)
    window_coef, window_metrics, window_scored = fit_probe(
        sample,
        WINDOW_FEATURES,
        label_col="oracle_unit_mse_gain",
        score_prefix="window_gate",
    )
    window_topk = topk_capture(
        window_scored,
        score_col="window_gate_score",
        label_col="oracle_unit_mse_gain",
        top_fracs=parse_float_list(args.top_fracs),
    )
    window_gain_coef, window_gain_metrics, window_gain_scored = fit_gain_regressors(
        sample,
        WINDOW_FEATURES,
        label_col="oracle_unit_mse_gain",
        score_prefix="window_gain",
    )
    window_gain_topk = pd.concat(
        [
            gain_topk_capture(
                window_gain_scored[window_gain_scored["gain_model"] == model].copy(),
                score_col=f"window_gain_{model}_pred_gain",
                label_col="oracle_unit_mse_gain",
                top_fracs=parse_float_list(args.top_fracs),
            )
            for model in sorted(window_gain_scored["gain_model"].unique())
        ],
        ignore_index=True,
    )
    window_gain_bins = pd.concat(
        [
            gain_quantile_bins(
                window_gain_scored[window_gain_scored["gain_model"] == model].copy(),
                score_col=f"window_gain_{model}_pred_gain",
                label_col="oracle_unit_mse_gain",
                energy_col="dynamic_energy",
            )
            for model in sorted(window_gain_scored["gain_model"].unique())
        ],
        ignore_index=True,
    )

    closed_loop_prefix = run_prefix(args.profile, args.closed_loop_tag)
    closed_loop_dir = Path(profile["out_dir"])
    schedule = read_one_csv(closed_loop_dir / f"{closed_loop_prefix}_closed_loop_schedule_selected.csv")
    lambda_cfg = read_one_csv(closed_loop_dir / f"{closed_loop_prefix}_closed_loop_lambda_selected.csv")
    lambda_splits = compute_selected_lambda_splits(
        profile,
        lambda_cfg=lambda_cfg,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
    )
    target_frames = []
    for split in ("val", "test"):
        target_frames.append(
            targetwise_rows(
                profile_name=args.profile,
                profile=profile,
                split=split,
                lambda_values=np.asarray(lambda_splits[split], dtype=np.float64),
                schedule=schedule,
                args=args,
            )
        )
    target = pd.concat(target_frames, ignore_index=True)
    target_coef, target_metrics, target_scored = fit_probe(
        target,
        TARGET_FEATURES,
        label_col="oracle_unit_mse_gain",
        score_prefix="target_gate",
    )
    target_topk = topk_capture(
        target_scored,
        score_col="target_gate_score",
        label_col="oracle_unit_mse_gain",
        top_fracs=parse_float_list(args.top_fracs),
    )
    target_gain_coef, target_gain_metrics, target_gain_scored = fit_gain_regressors(
        target,
        TARGET_FEATURES,
        label_col="oracle_unit_mse_gain",
        score_prefix="target_gain",
    )
    target_gain_topk = pd.concat(
        [
            gain_topk_capture(
                target_gain_scored[target_gain_scored["gain_model"] == model].copy(),
                score_col=f"target_gain_{model}_pred_gain",
                label_col="oracle_unit_mse_gain",
                top_fracs=parse_float_list(args.top_fracs),
            )
            for model in sorted(target_gain_scored["gain_model"].unique())
        ],
        ignore_index=True,
    )
    target_gain_bins = pd.concat(
        [
            gain_quantile_bins(
                target_gain_scored[target_gain_scored["gain_model"] == model].copy(),
                score_col=f"target_gain_{model}_pred_gain",
                label_col="oracle_unit_mse_gain",
                energy_col="dynamic_energy_target",
            )
            for model in sorted(target_gain_scored["gain_model"].unique())
        ],
        ignore_index=True,
    )
    target_by_var = target_summary(target_scored, score_col="target_gate_score")

    window_coef.to_csv(out_dir / f"{prefix}_window_coefficients.csv", index=False)
    window_metrics.to_csv(out_dir / f"{prefix}_window_metrics.csv", index=False)
    window_topk.to_csv(out_dir / f"{prefix}_window_topk_capture.csv", index=False)
    target_coef.to_csv(out_dir / f"{prefix}_target_coefficients.csv", index=False)
    target_metrics.to_csv(out_dir / f"{prefix}_target_metrics.csv", index=False)
    target_topk.to_csv(out_dir / f"{prefix}_target_topk_capture.csv", index=False)
    window_gain_coef.to_csv(out_dir / f"{prefix}_window_gain_coefficients.csv", index=False)
    window_gain_metrics.to_csv(out_dir / f"{prefix}_window_gain_metrics.csv", index=False)
    window_gain_topk.to_csv(out_dir / f"{prefix}_window_gain_topk_cvar.csv", index=False)
    window_gain_bins.to_csv(out_dir / f"{prefix}_window_gain_quantile_bins.csv", index=False)
    target_gain_coef.to_csv(out_dir / f"{prefix}_target_gain_coefficients.csv", index=False)
    target_gain_metrics.to_csv(out_dir / f"{prefix}_target_gain_metrics.csv", index=False)
    target_gain_topk.to_csv(out_dir / f"{prefix}_target_gain_topk_cvar.csv", index=False)
    target_gain_bins.to_csv(out_dir / f"{prefix}_target_gain_quantile_bins.csv", index=False)
    target_by_var.to_csv(out_dir / f"{prefix}_target_by_variable.csv", index=False)
    if args.write_target_scores:
        target_scored.to_csv(out_dir / f"{prefix}_target_scores.csv", index=False)
        target_gain_scored.to_csv(out_dir / f"{prefix}_target_gain_scores.csv", index=False)
    write_readme(
        out_dir,
        prefix,
        window_coef,
        window_metrics,
        target_coef,
        target_metrics,
        target_topk,
        window_gain_coef,
        window_gain_metrics,
        window_gain_topk,
        target_gain_coef,
        target_gain_metrics,
        target_gain_topk,
        window_gain_bins,
        target_gain_bins,
    )
    write_gain_figures(
        out_dir,
        prefix,
        window_gain_topk,
        target_gain_topk,
        window_gain_scored,
        target_gain_scored,
    )

    print("[Window metrics]", flush=True)
    print(window_metrics.to_string(index=False), flush=True)
    print("[Target metrics]", flush=True)
    print(target_metrics.to_string(index=False), flush=True)
    print("[Target coefficients]", flush=True)
    print(target_coef.to_string(index=False), flush=True)
    print("[Target gain metrics]", flush=True)
    print(target_gain_metrics.to_string(index=False), flush=True)
    print("[Target gain top-k / CVaR]", flush=True)
    print(target_gain_topk.to_string(index=False), flush=True)
    print("[Target gain quantile bins]", flush=True)
    print(target_gain_bins[target_gain_bins["split"] == "test"].to_string(index=False), flush=True)
    print(f"[Done] outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
