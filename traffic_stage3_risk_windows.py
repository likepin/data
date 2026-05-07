from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.diagnostics import transform_lambda_splits
from posthoc_calibration.profiles import PROFILES, compute_selected_lambda_splits
from posthoc_calibration.schedules import gamma_from_schedule
from traffic_stage3_lambda_three_source_pilot import (
    DEFAULT_CLOSED_LOOP_DIR,
    DEFAULT_STAGE2_DIR,
    build_target_masks,
    evaluate_selected_with_sample_stats,
    fold_ids,
    load_closed_loop_config,
    load_stage2_alpha,
    pct_gain,
)
from traffic_existing_prediction_ensemble import load_candidates


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
DEFAULT_STAGE3_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_stage3_lambda_three_source_closed_form_eta2"
DEFAULT_OUT_DIR = DATA_ROOT / "mechanism_evidence" / "traffic96_stage3_lambda_three_source_20260507" / "mechanism" / "risk_windows"
PROFILE_NAME = "traffic96_static"
STAGE2_PREFIX = "traffic96_static_stage2_light_seed2026"
CLOSED_LOOP_PREFIX = "traffic96_static_log_tail_quality_guard"
STAGE3_PREFIX = "traffic96_static_stage3_closed_form_eta2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Risk-window diagnostics for Traffic Stage3 closed-form eta2."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default=PROFILE_NAME)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--closed-loop-dir", type=Path, default=DEFAULT_CLOSED_LOOP_DIR)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap per split. 0 means full split.")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def read_one(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(df)}")
    return df.iloc[0]


def selected_spec(stage3_dir: Path) -> dict:
    row = read_one(stage3_dir / f"{STAGE3_PREFIX}_test_selected_summary.csv")
    return {
        "ensemble": str(row["ensemble"]),
        "eta_mode": str(row["eta_mode"]),
        "eta_mult": float(row["eta_mult"]),
        "eta_raw": float(row["eta_raw"]),
        "eta_clip_reason": str(row["eta_clip_reason"]),
        "target_mask": str(row["target_mask"]),
        "dynamic_source": str(row.get("dynamic_source", "static_p0")),
    }


def add_window_metrics(
    sample_stats: pd.DataFrame,
    *,
    split: str,
    lambda_values: np.ndarray,
    validation_folds: int,
) -> pd.DataFrame:
    df = sample_stats.copy()
    eta = df["eta_mult"].to_numpy(dtype=np.float64)
    gamma = df["gamma"].to_numpy(dtype=np.float64)
    anchor_sse = df["anchor_sse"].to_numpy(dtype=np.float64)
    err_dot_dyn = df["err_dot_dyn"].to_numpy(dtype=np.float64)
    dyn_sq = df["dyn_sq"].to_numpy(dtype=np.float64)
    count = df["count"].to_numpy(dtype=np.float64)
    stage3_sse = anchor_sse - 2.0 * eta * gamma * err_dot_dyn + np.square(eta * gamma) * dyn_sq

    n_samples = len(df)
    if len(lambda_values) < n_samples:
        raise RuntimeError(f"Lambda length mismatch for {split}: lambda={len(lambda_values)} required={n_samples}")
    df["lambda_value"] = np.asarray(lambda_values[:n_samples], dtype=np.float64)
    df["fold"] = fold_ids(n_samples, validation_folds)
    df["anchor_mse"] = anchor_sse / count
    df["stage3_sse"] = stage3_sse
    df["stage3_mse"] = stage3_sse / count
    df["mse_gain"] = df["anchor_mse"] - df["stage3_mse"]
    df["mse_gain_pct"] = np.where(
        np.abs(df["anchor_mse"]) > 1e-12,
        100.0 * df["mse_gain"] / df["anchor_mse"],
        0.0,
    )
    df["sse_gain"] = anchor_sse - stage3_sse
    df["gamma_rank_pct"] = df["gamma"].rank(method="average", pct=True)
    df["gamma_ordinal_rank_pct"] = df["gamma"].rank(method="first", pct=True)
    df["lambda_rank_pct"] = df["lambda_value"].rank(method="average", pct=True)
    return df


def rank_fraction_mask(df: pd.DataFrame, fraction: float, highest: bool) -> pd.Series:
    n = max(1, int(np.ceil(len(df) * fraction)))
    ordered = df.sort_values(["gamma", "sample_id"], ascending=[not highest, True])
    chosen = ordered.head(n).index
    mask = pd.Series(False, index=df.index)
    mask.loc[chosen] = True
    return mask


def subset_mask(df: pd.DataFrame, name: str) -> pd.Series:
    gamma_min = float(df["gamma"].min())
    gamma_max = float(df["gamma"].max())
    eps = 1e-12
    if name == "all":
        return pd.Series(True, index=df.index)
    if name == "gamma_floor":
        return df["gamma"] <= gamma_min + eps
    if name == "gamma_active_gt_floor":
        return df["gamma"] > gamma_min + eps
    if name == "gamma_max_saturated":
        return df["gamma"] >= gamma_max - eps
    if name == "top_rank_1pct_gamma":
        return rank_fraction_mask(df, 0.01, highest=True)
    if name == "top_rank_5pct_gamma":
        return rank_fraction_mask(df, 0.05, highest=True)
    if name == "top_rank_10pct_gamma":
        return rank_fraction_mask(df, 0.10, highest=True)
    if name == "bottom_rank_10pct_gamma":
        return rank_fraction_mask(df, 0.10, highest=False)
    raise ValueError(f"Unknown risk group: {name}")


def summarize_group(df: pd.DataFrame, split: str, group: str) -> dict:
    mask = subset_mask(df, group)
    sub = df[mask].copy()
    if sub.empty:
        return {
            "split": split,
            "risk_group": group,
            "n_windows": 0,
            "coverage_pct": 0.0,
            "gamma_min": np.nan,
            "gamma_mean": np.nan,
            "gamma_max": np.nan,
            "lambda_mean": np.nan,
            "anchor_mse": np.nan,
            "stage3_mse": np.nan,
            "mse_gain_pct": np.nan,
            "sse_gain_share_pct": np.nan,
            "positive_window_fraction": np.nan,
        }
    anchor_sse = float(sub["anchor_sse"].sum())
    stage3_sse = float(sub["stage3_sse"].sum())
    count = float(sub["count"].sum())
    total_sse_gain = float((df["anchor_sse"] - df["stage3_sse"]).sum())
    group_sse_gain = anchor_sse - stage3_sse
    share = 100.0 * group_sse_gain / total_sse_gain if abs(total_sse_gain) > 1e-12 else 0.0
    return {
        "split": split,
        "risk_group": group,
        "n_windows": int(len(sub)),
        "coverage_pct": 100.0 * len(sub) / len(df),
        "gamma_min": float(sub["gamma"].min()),
        "gamma_mean": float(sub["gamma"].mean()),
        "gamma_max": float(sub["gamma"].max()),
        "lambda_mean": float(sub["lambda_value"].mean()),
        "anchor_mse": anchor_sse / count,
        "stage3_mse": stage3_sse / count,
        "mse_gain_pct": pct_gain(anchor_sse / count, stage3_sse / count),
        "sse_gain_share_pct": share,
        "positive_window_fraction": float((sub["sse_gain"] > 0).mean()),
    }


def risk_group_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    groups = [
        "all",
        "gamma_floor",
        "gamma_active_gt_floor",
        "gamma_max_saturated",
        "top_rank_1pct_gamma",
        "top_rank_5pct_gamma",
        "top_rank_10pct_gamma",
        "bottom_rank_10pct_gamma",
    ]
    return pd.DataFrame([summarize_group(df, split, group) for group in groups])


def fold_contribution_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    total_sse_gain = float(df["sse_gain"].sum())
    for fold, sub in df.groupby("fold", sort=True):
        anchor_sse = float(sub["anchor_sse"].sum())
        stage3_sse = float(sub["stage3_sse"].sum())
        sse_gain = anchor_sse - stage3_sse
        rows.append(
            {
                "split": split,
                "fold": int(fold),
                "n_windows": int(len(sub)),
                "gamma_mean": float(sub["gamma"].mean()),
                "gamma_p90": float(sub["gamma"].quantile(0.90)),
                "anchor_mse": anchor_sse / float(sub["count"].sum()),
                "stage3_mse": stage3_sse / float(sub["count"].sum()),
                "mse_gain_pct": pct_gain(anchor_sse / float(sub["count"].sum()), stage3_sse / float(sub["count"].sum())),
                "sse_gain_share_pct": 100.0 * sse_gain / total_sse_gain if abs(total_sse_gain) > 1e-12 else 0.0,
                "positive_window_fraction": float((sub["sse_gain"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def top_windows(df: pd.DataFrame, split: str, top_k: int) -> pd.DataFrame:
    columns = [
        "split",
        "sample_id",
        "fold",
        "gamma",
        "gamma_rank_pct",
        "lambda_value",
        "lambda_rank_pct",
        "gamma_ordinal_rank_pct",
        "anchor_mse",
        "stage3_mse",
        "mse_gain",
        "mse_gain_pct",
        "sse_gain",
        "err_dot_dyn",
        "dyn_sq",
    ]
    best = df.sort_values(["sse_gain", "gamma"], ascending=[False, False]).head(top_k).copy()
    best["rank_type"] = "best_sse_gain"
    high_risk = df.sort_values(["gamma", "sse_gain"], ascending=[False, False]).head(top_k).copy()
    high_risk["rank_type"] = "highest_gamma"
    out = pd.concat([best, high_risk], ignore_index=True)
    return out[["rank_type", *columns]]


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(risk: pd.DataFrame, folds: pd.DataFrame, out_dir: Path) -> None:
    def row(split: str, group: str) -> pd.Series:
        rows = risk[(risk["split"] == split) & (risk["risk_group"] == group)]
        if len(rows) != 1:
            raise ValueError(f"Expected one risk row for {split}/{group}, got {len(rows)}")
        return rows.iloc[0]

    test_all = row("test", "all")
    test_floor = row("test", "gamma_floor")
    test_active = row("test", "gamma_active_gt_floor")
    test_top5 = row("test", "top_rank_5pct_gamma")
    val_fold4 = folds[(folds["split"] == "val") & (folds["fold"] == 4)].iloc[0]
    lines = [
        "# Traffic96 Stage3 Risk Windows",
        "",
        "Scope: Stage3 closed-form eta2, with eta and target mask fixed before this diagnostic.",
        "",
        "Key test observations:",
        f"- Overall test MSE gain vs Stage2 anchor: `{float(test_all['mse_gain_pct']):.4f}%`.",
        f"- `gamma_floor` covers `{float(test_floor['coverage_pct']):.2f}%` of test windows and contributes `{float(test_floor['sse_gain_share_pct']):.2f}%` of total SSE gain.",
        f"- `gamma_active_gt_floor` covers `{float(test_active['coverage_pct']):.2f}%` of test windows but has MSE gain `{float(test_active['mse_gain_pct']):.4f}%`.",
        f"- `top_rank_5pct_gamma` has MSE gain `{float(test_top5['mse_gain_pct']):.4f}%`.",
        "",
        "Validation observation:",
        f"- Validation Fold 4 remains the strongest fold-level gain region: `{float(val_fold4['mse_gain_pct']):.4f}%` MSE gain.",
        "",
        "Interpretation:",
        "- The current Stage3 eta2 result should not be framed as high-risk-window localization.",
        "- Test improvement is mostly a weak global/floor correction effect; high-gamma active windows are still unstable.",
        "- This supports keeping the guard narrative: lambda is useful as a diagnostic signal, but current dynamic correction is not yet a reliable high-risk attack module.",
        "",
        "Files:",
        "- `traffic96_stage3_eta2_risk_group_table.csv`: risk bucket metrics.",
        "- `traffic96_stage3_eta2_fold_contribution.csv`: fold-level contribution metrics.",
        "- `traffic96_stage3_eta2_top_risk_windows.csv`: best-gain and highest-gamma window lists.",
        "- `traffic96_stage3_eta2_{val,test}_risk_windows_sample_stats.csv`: sample-level diagnostic table.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.profile != PROFILE_NAME:
        raise ValueError("Risk-window diagnostics are currently scoped to traffic96_static.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    profile = dict(PROFILES[args.profile])
    alpha = load_stage2_alpha(args.stage2_dir, STAGE2_PREFIX)
    spec = selected_spec(args.stage3_dir)
    target_masks = build_target_masks([spec["target_mask"]], alpha)
    lambda_cfg, schedule = load_closed_loop_config(args.closed_loop_dir, CLOSED_LOOP_PREFIX)
    raw_lambda_splits = compute_selected_lambda_splits(
        profile,
        lambda_cfg=lambda_cfg,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
    )
    lambda_splits = transform_lambda_splits(raw_lambda_splits, lambda_cfg.get("lambda_transform", "raw"))
    gamma_splits = {
        split: gamma_from_schedule(values, schedule).astype(np.float32)
        for split, values in lambda_splits.items()
    }
    candidates = load_candidates(profile)
    interface_dir = Path(profile["interface_dir"])

    manifest = {
        "profile": args.profile,
        "stage3_dir": str(args.stage3_dir),
        "stage2_dir": str(args.stage2_dir),
        "closed_loop_dir": str(args.closed_loop_dir),
        "interface_dir": str(interface_dir),
        "selected_spec": spec,
        "lambda_cfg": lambda_cfg,
        "schedule": schedule,
        "max_samples": args.max_samples,
        "top_k": args.top_k,
        "diagnostic_scope": (
            "Post-selection mechanism diagnostic only. Eta and target mask are fixed from validation-selected "
            "Stage3 closed-form eta2; test is not used for selection."
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_risk_rows = []
    all_fold_rows = []
    all_top_rows = []
    summaries = []
    for split in ["val", "test"]:
        summary, sample_stats = evaluate_selected_with_sample_stats(
            split=split,
            candidates=candidates,
            alpha=alpha,
            delta_path=interface_dir / f"deltaA_{split}.npy",
            gamma=gamma_splits[split],
            spec=spec,
            target_masks=target_masks,
            dynamic_source=spec["dynamic_source"],
            chunk_size=args.chunk_size,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
        )
        enriched = add_window_metrics(
            sample_stats,
            split=split,
            lambda_values=lambda_splits[split],
            validation_folds=args.validation_folds,
        )
        enriched.to_csv(args.out_dir / f"traffic96_stage3_eta2_{split}_risk_windows_sample_stats.csv", index=False)
        summaries.append(summary)
        all_risk_rows.append(risk_group_table(enriched, split))
        all_fold_rows.append(fold_contribution_table(enriched, split))
        all_top_rows.append(top_windows(enriched, split, args.top_k))

    risk = pd.concat(all_risk_rows, ignore_index=True)
    folds = pd.concat(all_fold_rows, ignore_index=True)
    top = pd.concat(all_top_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    risk.to_csv(args.out_dir / "traffic96_stage3_eta2_risk_group_table.csv", index=False)
    folds.to_csv(args.out_dir / "traffic96_stage3_eta2_fold_contribution.csv", index=False)
    top.to_csv(args.out_dir / "traffic96_stage3_eta2_top_risk_windows.csv", index=False)
    summary_df.to_csv(args.out_dir / "traffic96_stage3_eta2_recomputed_summary.csv", index=False)
    write_markdown_table(risk, args.out_dir / "traffic96_stage3_eta2_risk_group_table.md")
    write_markdown_table(folds, args.out_dir / "traffic96_stage3_eta2_fold_contribution.md")
    write_readme(risk, folds, args.out_dir)

    test_risk = risk[
        (risk["split"] == "test")
        & (risk["risk_group"].isin(["all", "gamma_floor", "gamma_active_gt_floor", "top_rank_5pct_gamma"]))
    ]
    print(test_risk.to_string(index=False), flush=True)
    print(f"[Wrote] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
